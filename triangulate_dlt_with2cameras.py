# triangulate_dlt_with2cameras.py
# İki kamera (Cam1 & Cam2) ile DLT triangulation + robust refine

import numpy as np
import cv2
import json
import configparser
import xml.etree.ElementTree as ET
from statistics import median

# ---------------- Ayarlar ----------------
REPORT_PREFIX   = "2cams"
MAX_OFF2        = 10          # Cam2 ofset aralığı (kaba arama)
MAX_EVAL_FRAMES = 400         # skor değerlendirme uzunluğu
HUBER_DELTA     = 5.0         # Huber eşiği (px)
LM_ITERS        = 10          # refine iter sayısı
ORDERS = ['xyz','xzy','yxz','yzx','zxy','zyx']

# ----------- Dosya eşlemesi (projene göre) -----------
cam_ids   = [38738369, 31118929]  # Cam1, Cam2
und_files = [
    "undistorted_points_v2/38738369_undistorted_points_v2.txt",
    "undistorted_points_v2/31118929_undistorted_points_v2.txt",
]
conf_files = [
    "configuration_files/SN38738369.conf",
    "configuration_files/SN31118929.conf",
]
xml_files  = [
    "xml_files/38738369.xml",
    "xml_files/31118929.xml",
]
json_path  = "2025_03_27-1034.json"

# ---------------- Yardımcılar ----------------
def get_cam_key_from_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    src  = (root.findtext(".//source") or "")
    W = int(root.findtext(".//original_size/width"))
    H = int(root.findtext(".//original_size/height"))
    side = "LEFT" if ("leftView" in src) else "RIGHT"
    if   (W,H)==(1280,720):  res="HD"
    elif (W,H)==(1920,1080): res="FHD"
    else:                    res="2K"
    return f"{side}_CAM_{res}"

def load_K(conf_path, cam_key):
    cfg=configparser.ConfigParser(); cfg.read(conf_path)
    cam=cfg[cam_key]
    fx,fy = float(cam['fx']), float(cam['fy'])
    cx,cy = float(cam['cx']), float(cam['cy'])
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], float)

def euler_to_R(rx, ry, rz, order='xyz'):
    cx, cy, cz = np.cos([rx,ry,rz]); sx, sy, sz = np.sin([rx,ry,rz])
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    maps={
        'xyz': Rz@Ry@Rx, 'xzy': Ry@Rz@Rx,
        'yxz': Rz@Rx@Ry, 'yzx': Rx@Rz@Ry,
        'zxy': Ry@Rx@Rz, 'zyx': Rx@Ry@Rz,
    }
    return maps[order]

def load_extrinsics_world_to_camera(json_path, cam_id, order='xyz'):
    data=json.load(open(json_path,"r",encoding="utf-8"))
    rw=np.array(data[str(cam_id)]['world']['rotation'],float)        # rad
    C =np.array(data[str(cam_id)]['world']['translation'],float).reshape(3,1) # m
    R_cw=euler_to_R(rw[0],rw[1],rw[2],order=order)  # camera->world
    R = R_cw.T
    t = -R @ C
    return R, t

def build_P(K, R, t):
    return K @ np.hstack([R,t])

def load_points_txt(path):
    d={}
    with open(path,"r",encoding="utf-8") as f:
        for L in f:
            s=L.split()
            if len(s)<3: continue
            d[int(s[0])] = (float(s[1]), float(s[2]))
    return d

def shift_frames_dict(d, off):
    return d if off==0 else {f+off:xy for f,xy in d.items()}

def common_frames(*dicts):
    keys=set(dicts[0].keys())
    for d in dicts[1:]: keys &= set(d.keys())
    return sorted(keys)

# ------------- DLT / Refine -------------
def triangulate_dlt(Ps, xys):
    A=[]
    for P,(x,y) in zip(Ps, xys):
        A.append(x*P[2]-P[0]); A.append(y*P[2]-P[1])
    A=np.asarray(A)
    _,_,Vt=np.linalg.svd(A)
    Xh=Vt[-1]; Xh/=Xh[3]
    return Xh[:3]

def proj_and_jac(P, X):
    Xh=np.append(X,1.0)
    p1,p2,p3=P[0],P[1],P[2]
    u_num=p1@Xh; v_num=p2@Xh; w=p3@Xh
    u=u_num/w; v=v_num/w
    du=(p1[:3]*w - u*p3[:3])/(w*w)
    dv=(p2[:3]*w - v*p3[:3])/(w*w)
    return np.array([u,v]), np.vstack([du,dv])

def huber_weights(r, delta):
    a=np.abs(r); w=np.ones_like(a); m=a>delta; w[m]=delta/a[m]
    return w

def refine_point(Ps, xys, X0, iters=10, delta=5.0):
    X=X0.copy()
    for _ in range(iters):
        J_list=[]; r_list=[]
        for P,(x,y) in zip(Ps, xys):
            (u,v),J=proj_and_jac(P,X)
            r=np.array([x-u, y-v])
            J_list.append(J); r_list.append(r)
        J=np.vstack(J_list); r=np.hstack(r_list)
        W=np.diag(huber_weights(r, delta))
        JT_W=J.T@W
        H=JT_W@J; b=JT_W@r
        try:
            dX=np.linalg.lstsq(H,b,rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        X+=dX
        if np.linalg.norm(dX)<1e-6: break
    errs=[]
    for P,(x,y) in zip(Ps, xys):
        (u,v),_=proj_and_jac(P,X)
        errs.append(float(np.hypot(x-u,y-v)))
    return X, errs

def safe_stats(values):
    a = np.asarray(values, float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return None
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "max": float(a.max()),
        "N": int(a.size),
    }

# ---------------- Ana akış ----------------
if __name__=="__main__":
    # Intrinsics
    Ks=[]
    for conf,xml in zip(conf_files, xml_files):
        Ks.append(load_K(conf, get_cam_key_from_xml(xml)))

    # 2D noktalar
    PTS=[load_points_txt(p) for p in und_files]

    # ------- Kaba arama: Euler sırası + cam2 ofset -------
    best={"order":None,"off2":0,"score":1e18,"frames_used":0}
    for order in ORDERS:
        P1=build_P(Ks[0], *load_extrinsics_world_to_camera(json_path, cam_ids[0], order))
        P2=build_P(Ks[1], *load_extrinsics_world_to_camera(json_path, cam_ids[1], order))
        for off2 in range(-MAX_OFF2, MAX_OFF2+1):
            D1=PTS[0]; D2=shift_frames_dict(PTS[1], off2)
            frs=common_frames(D1,D2)
            if len(frs)<10: continue
            N=min(MAX_EVAL_FRAMES,len(frs))
            total=0.0; cnt=0
            for f in frs[:N]:
                X0=triangulate_dlt([P1,P2],[D1[f],D2[f]])
                for P,xy in zip([P1,P2],[D1[f],D2[f]]):
                    Xh=np.append(X0,1.0); uvw=P@Xh
                    u,v=uvw[0]/uvw[2], uvw[1]/uvw[2]
                    total += float(np.hypot(xy[0]-u, xy[1]-v)); cnt += 1
            score=total/max(cnt,1)
            if score<best["score"]:
                best.update({"order":order,"off2":off2,"score":score,"frames_used":N})

    # ------- Final: iki kamera ile triangulation -------
    P1=build_P(Ks[0], *load_extrinsics_world_to_camera(json_path, cam_ids[0], best["order"]))
    P2=build_P(Ks[1], *load_extrinsics_world_to_camera(json_path, cam_ids[1], best["order"]))
    D1=PTS[0]; D2=shift_frames_dict(PTS[1], best["off2"])
    frames_all=common_frames(D1,D2)

    out3d=open(f"triangulated_points_{REPORT_PREFIX}.txt","w",encoding="utf-8")
    ser1=open(f"reprojection_series_{REPORT_PREFIX}_cam1.txt","w",encoding="utf-8")
    ser2=open(f"reprojection_series_{REPORT_PREFIX}_cam2.txt","w",encoding="utf-8")

    e1_list=[]; e2_list=[]

    for f in frames_all:
        xys=[D1[f], D2[f]]
        X0=triangulate_dlt([P1,P2], xys)
        Xr, errs = refine_point([P1,P2], xys, X0, iters=LM_ITERS, delta=HUBER_DELTA)
        out3d.write(f"{f} {Xr[0]:.6f} {Xr[1]:.6f} {Xr[2]:.6f}\n")
        ser1.write(f"{f} {errs[0]:.3f}\n")
        ser2.write(f"{f} {errs[1]:.3f}\n")
        e1_list.append(errs[0]); e2_list.append(errs[1])

    out3d.close(); ser1.close(); ser2.close()

    # İstatistik
    s1=safe_stats(e1_list); s2=safe_stats(e2_list)
    with open(f"reprojection_stats_{REPORT_PREFIX}.txt","w",encoding="utf-8") as fr:
        if s1 is None: fr.write("Cam1: N=0\n")
        else: fr.write(f"Cam1: mean={s1['mean']:.3f} px, median={s1['median']:.3f} px, max={s1['max']:.3f} px, N={s1['N']}\n")
        if s2 is None: fr.write("Cam2: N=0\n")
        else: fr.write(f"Cam2: mean={s2['mean']:.3f} px, median={s2['median']:.3f} px, max={s2['max']:.3f} px, N={s2['N']}\n")

    with open(f"chosen_params_{REPORT_PREFIX}.txt","w",encoding="utf-8") as fc:
        fc.write(f"Best Euler order = {best['order']}\n")
        fc.write(f"Best frame offsets: cam2={best['off2']} (cam1=0)\n")
        fc.write(f"Coarse search score (cam1-2) ~= {best['score']:.3f} px on first {best['frames_used']} frames\n")

    print("[OK] triangulated_points_2cams.txt ve rapor dosyalari yazildi.")
