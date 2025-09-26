# v5: İki-ankorlu (Cam1 & Cam2) triangülasyon + Cam3 opsiyonel
# - Kaba arama: sadece Cam1-2 ile (Euler order + Cam2 ofset)
# - Cam3 için tek boyutlu global ofset taraması (skor: 2-cam çözümün Cam3 reproj hatası)
# - Her framede 2-cam robust refine; Cam3 hata < eşik ise refine'e dahil edilir
# - Huber ağırlık, UTF-8 dosya yazımı, kapsamlı raporlar

import numpy as np
import cv2
import json
import configparser
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import median

# ------------- Ayarlar -------------
MAX_OFF2 = 10          # Cam2 ofset aralığı (kaba arama)
MAX_OFF3 = 10          # Cam3 ofset aralığı (tek eksen arama)
MAX_EVAL_FRAMES = 400  # skor için kullanılan çerçeve sayısı
HUBER_DELTA = 5.0      # robust delta (px)
LM_ITERS = 10          # refine iter sayısı
CAM3_USE_THRESH = 20.0 # Cam3'ü dahil etme eşiği (px). Büyükse dahil ETME.
REPORT_PREFIX = "v5"

# ---------- IO & Geometri ----------

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
    rw=np.array(data[str(cam_id)]['world']['rotation'],float) # rad
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

# --------- Triangulation/Refine ---------

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
    # per-camera reproj
    errs=[]
    for P,(x,y) in zip(Ps, xys):
        (u,v),_=proj_and_jac(P,X)
        errs.append(float(np.hypot(x-u,y-v)))
    return X, errs

# --------------- Ana Akış ---------------

if __name__=="__main__":
    # Senin dosya eşlemen
    cam_ids=[38738369, 31118929, 38838483]
    und_files=[
        "undistorted_points_v2/38738369_undistorted_points_v2.txt",
        "undistorted_points_v2/31118929_undistorted_points_v2.txt",
        "undistorted_points_v2/38838483_undistorted_points_v2.txt",
    ]
    conf_files=[
        "configuration_files/SN38738369.conf",
        "configuration_files/SN31118929.conf",
        "configuration_files/SN38838483.conf",
    ]
    xml_files=[
        "xml_files/38738369.xml",
        "xml_files/31118929.xml",
        "xml_files/38838483.xml",
    ]
    json_path="2025_03_27-1034.json"

    # Intrinsics
    Ks=[]
    for conf,xml in zip(conf_files, xml_files):
        Ks.append(load_K(conf, get_cam_key_from_xml(xml)))

    # Extrinsics için denenebilecek Euler sıraları
    ORDERS=['xyz','xzy','yxz','yzx','zxy','zyx']

    # 2D noktalar
    PTS=[load_points_txt(p) for p in und_files]

    # ------- Kaba arama: sadece Cam1-2 -------
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

    # ------- Cam3 için tek boyutlu ofset araması -------
    # 2-cam çözüm üzerinden Cam3'ün medyan hatasını minimize et
    P1=build_P(Ks[0], *load_extrinsics_world_to_camera(json_path, cam_ids[0], best["order"]))
    P2=build_P(Ks[1], *load_extrinsics_world_to_camera(json_path, cam_ids[1], best["order"]))
    D1=PTS[0]; D2=shift_frames_dict(PTS[1], best["off2"])
    fr12=common_frames(D1,D2)

    best_off3=0; best_med=1e18
    for off3 in range(-MAX_OFF3, MAX_OFF3+1):
        D3=shift_frames_dict(PTS[2], off3)
        frs=sorted(set(fr12) & set(D3.keys()))
        if len(frs)<20: continue
        errs=[]
        for f in frs[:MAX_EVAL_FRAMES]:
            X0=triangulate_dlt([P1,P2],[D1[f],D2[f]])
            # Cam3'ün reproj hatası
            P3=build_P(Ks[2], *load_extrinsics_world_to_camera(json_path, cam_ids[2], best["order"]))
            Xh=np.append(X0,1.0); uvw=P3@Xh
            u,v=uvw[0]/uvw[2], uvw[1]/uvw[2]
            x3,y3=D3[f]
            errs.append(float(np.hypot(x3-u,y3-v)))
        if errs:
            m=median(errs)
            if m<best_med:
                best_med=m; best_off3=off3

    # ------- Final çözüm: 2-cam temel + Cam3 eşiğe bağlı -------
    P3=build_P(Ks[2], *load_extrinsics_world_to_camera(json_path, cam_ids[2], best["order"]))
    D3=shift_frames_dict(PTS[2], best_off3)

    frames_all=fr12  # iki anchor ortak çerçeveler
    out3d=open(f"triangulated_points_3cams_{REPORT_PREFIX}.txt","w",encoding="utf-8")
    ser_cam=[open(f"reprojection_series_{REPORT_PREFIX}_cam{i+1}.txt","w",encoding="utf-8") for i in range(3)]
    errs_per_cam=[[],[],[]]; used_cam3=0

    for f in frames_all:
        xys12=[D1[f], D2[f]]
        # 2-cam DLT + refine (temel)
        X0=triangulate_dlt([P1,P2], xys12)
        Xr, e12 = refine_point([P1,P2], xys12, X0, iters=LM_ITERS, delta=HUBER_DELTA)

        # Cam3 varsa ve hatası küçükse dahil et
        if f in D3:
            (u3v3,_J)=proj_and_jac(P3,Xr)
            u3,v3=u3v3
            e3_pred=float(np.hypot(D3[f][0]-u3, D3[f][1]-v3))
            if e3_pred < CAM3_USE_THRESH:
                Xr, e123 = refine_point([P1,P2,P3], [D1[f],D2[f],D3[f]], Xr,
                                        iters=LM_ITERS, delta=HUBER_DELTA)
                used_cam3 += 1
                e12 = e123[:2]
                real_e3 = e123[2]
            else:
                # Cam3 kullanılmadı; yine de rapor için reproj yaz
                Xh=np.append(Xr,1.0); uvw=P3@Xh
                u3,v3=uvw[0]/uvw[2], uvw[1]/uvw[2]
                real_e3=float(np.hypot(D3[f][0]-u3, D3[f][1]-v3))
        else:
            real_e3 = np.nan

        # rapor
        out3d.write(f"{f} {Xr[0]:.6f} {Xr[1]:.6f} {Xr[2]:.6f}\n")
        # Cam1-2 hataları:
        # e12 listesi 2 elemanlı; sırayla yaz
        ser_cam[0].write(f"{f} {e12[0]:.3f}\n")
        ser_cam[1].write(f"{f} {e12[1]:.3f}\n")
        ser_cam[2].write(f"{f} {real_e3:.3f}\n")

        errs_per_cam[0].append(e12[0])
        errs_per_cam[1].append(e12[1])
        errs_per_cam[2].append(real_e3 if not np.isnan(real_e3) else np.nan)

    out3d.close()
    for fh in ser_cam: fh.close()

    # İstatistik (NaN'leri at)
    with open(f"reprojection_stats_{REPORT_PREFIX}.txt","w",encoding="utf-8") as fr:
        for i,arr in enumerate(errs_per_cam, start=1):
            a=np.array(arr, float)
            a=a[~np.isnan(a)]
            fr.write(f"Cam{i}: mean={a.mean():.3f} px, median={median(a):.3f} px, max={a.max():.3f} px, N={len(a)}\n")

    with open(f"chosen_params_{REPORT_PREFIX}.txt","w",encoding="utf-8") as fc:
        fc.write(f"Best Euler order = {best['order']}\n")
        fc.write(f"Best frame offsets: cam2={best['off2']}, cam3={best_off3} (cam1=0)\n")
        fc.write(f"Coarse search score (cam1-2) ~= {best['score']:.3f} px on first {best['frames_used']} frames\n")
        fc.write(f"CAM3_USE_THRESH = {CAM3_USE_THRESH} px, used_cam3 = {used_cam3} / {len(frames_all)} frames\n")

    print(f"[OK] triangulated_points_3cams_{REPORT_PREFIX}.txt ve rapor dosyaları yazıldı.")
