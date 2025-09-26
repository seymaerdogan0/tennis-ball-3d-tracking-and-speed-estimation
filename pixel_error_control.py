import numpy as np, json, configparser, xml.etree.ElementTree as ET
from statistics import median

# --------- Girdi dosyaları ---------
TRI_FILE = "triangulated_points_3cams_v5.txt"    # v5b ise dosya adını değiştir
UND = [
    "undistorted_points_v2/38738369_undistorted_points_v2.txt",
    "undistorted_points_v2/31118929_undistorted_points_v2.txt",
    "undistorted_points_v2/38838483_undistorted_points_v2.txt",
]
CONF = [
    "configuration_files/SN38738369.conf",
    "configuration_files/SN31118929.conf",
    "configuration_files/SN38838483.conf",
]
XML  = ["xml_files/38738369.xml","xml_files/31118929.xml","xml_files/38838483.xml"]
JSON_PATH = "2025_03_27-1034.json"

# chosen_params_*.txt içinden Euler ve ofsetleri çek
def read_chosen(prefix_guess_list=("v5b","v5","v9")):
    order = "xyz"; off2 = 0; off3 = 0
    for p in prefix_guess_list:
        try:
            with open(f"chosen_params_{p}.txt","r",encoding="utf-8") as f:
                txt = f.read()
            # "Best Euler order = yzx"
            for L in txt.splitlines():
                if "Best Euler order" in L:
                    order = L.split("=")[1].strip()
                if "Best frame offsets" in L:
                    # "cam2=..., cam3=... (cam1=0)"
                    parts = L.split(":")[1].split("(")[0]
                    for kv in parts.split(","):
                        k,v = kv.strip().split("=")
                        if k=="cam2": off2 = int(v)
                        if k=="cam3": off3 = int(v)
            return order, off2, off3, p
        except FileNotFoundError:
            continue
    return order, off2, off3, None

EULER_ORDER, OFF2, OFF3, CHOSEN_SRC = read_chosen()

# ---------- Yardımcılar ----------
def get_cam_key_from_xml(xml_path):
    r=ET.parse(xml_path).getroot()
    side = "LEFT" if "leftView" in (r.findtext(".//source") or "") else "RIGHT"
    W=int(r.findtext(".//original_size/width")); H=int(r.findtext(".//original_size/height"))
    res = "HD" if (W,H)==(1280,720) else ("FHD" if (W,H)==(1920,1080) else "2K")
    return f"{side}_CAM_{res}"

def load_K(conf_path, cam_key):
    cfg=configparser.ConfigParser(); cfg.read(conf_path)
    c=cfg[cam_key]
    fx,fy = float(c['fx']), float(c['fy'])
    cx,cy = float(c['cx']), float(c['cy'])
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], float)

def euler_to_R(rx,ry,rz,order):
    cx,cy,cz=np.cos([rx,ry,rz]); sx,sy,sz=np.sin([rx,ry,rz])
    Rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    maps={'xyz':Rz@Ry@Rx, 'xzy':Ry@Rz@Rx, 'yxz':Rz@Rx@Ry, 'yzx':Rx@Rz@Ry, 'zxy':Ry@Rx@Rz, 'zyx':Rx@Ry@Rz}
    return maps[order]

def load_RT(json_path, cam_id, order):
    d=json.load(open(json_path,"r",encoding="utf-8"))
    r=np.array(d[str(cam_id)]['world']['rotation'],float)   # rad
    C=np.array(d[str(cam_id)]['world']['translation'],float).reshape(3,1)
    Rcw=euler_to_R(r[0],r[1],r[2],order)  # camera->world
    R = Rcw.T
    t = -R @ C                             # world->camera
    return R, t

def load_2d(path):
    out={}
    for L in open(path,encoding="utf-8"):
        p=L.split()
        if len(p)>=3:
            out[int(p[0])] = (float(p[1]), float(p[2]))
    return out

def shift_dict(d, off):
    if off==0: return d
    return {f+off:xy for f,xy in d.items()}

def read_3d(path):
    F=[]; X=[]
    for L in open(path,encoding="utf-8"):
        p=L.split()
        if len(p)>=4:
            F.append(int(p[0])); X.append([float(p[1]),float(p[2]),float(p[3])])
    return np.array(F), np.array(X,float)

def project_uv(K, Xc):
    fx,fy = K[0,0], K[1,1]; cx,cy = K[0,2], K[1,2]
    Z = Xc[2] if Xc[2]!=0 else 1e-9
    u = fx*(Xc[0]/Z) + cx
    v = fy*(Xc[1]/Z) + cy
    return u,v

# ---------- Yükle ----------
cams=[38738369,31118929,38838483]
Ks=[load_K(c, get_cam_key_from_xml(x)) for c,x in zip(CONF,XML)]
RT=[load_RT(JSON_PATH,c,EULER_ORDER) for c in cams]
D2=[load_2d(p) for p in UND]
# V5/V5b ile aynı frame senkronu:
D2[1] = shift_dict(D2[1], OFF2)
D2[2] = shift_dict(D2[2], OFF3)
F3, X3 = read_3d(TRI_FILE)

# ---------- Hesapla ----------
for i in range(3):
    R,t = RT[i]; K = Ks[i]; fx = K[0,0]
    px_errs=[]; cm_errs=[]; cm_per_px_list=[]
    for f, Xw in zip(F3, X3):
        if f not in D2[i]: continue
        Xc = (R @ Xw.reshape(3,1) + t).ravel()
        u,v = project_uv(K, Xc)
        ex = D2[i][f][0] - u; ey = D2[i][f][1] - v
        epx = float(np.hypot(ex,ey))
        px_errs.append(epx)
        cm_per_px = 100.0 * abs(Xc[2]) / fx
        cm_per_px_list.append(cm_per_px)
        cm_errs.append(epx * cm_per_px)
    if px_errs:
        a=np.array(px_errs); b=np.array(cm_errs); c=np.array(cm_per_px_list)
        print(f"Cam{i+1}:  px -> mean={a.mean():.3f}, median={np.median(a):.3f}, max={a.max():.3f}, N={a.size}")
        print(f"        cm -> mean={b.mean():.2f}, median={np.median(b):.2f}, max={b.max():.2f}")
        print(f"        tipik ölçek: median(cm/px) = {np.median(c):.2f}  (1 px ≈ bu kadar cm)\n")
    else:
        print(f"Cam{i+1}: veri yok\n")

src = CHOSEN_SRC or "(manuel varsayılan)"
print(f"[Kullanılan Euler order] {EULER_ORDER}  |  [Ofsetler] cam2={OFF2}, cam3={OFF3}  |  Kaynak: {src}")
print(f"[Not] 3D dosyası olarak ölçeklenmemiş '{TRI_FILE}' kullanın (v5/v5b).")
