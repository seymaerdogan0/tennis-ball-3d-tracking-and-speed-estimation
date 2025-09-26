# triangulate_dlt_v1_with_reproj_stats.py
import numpy as np
import cv2
import json
import configparser
import xml.etree.ElementTree as ET
from statistics import median

def get_cam_key_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    source = root.find(".//source")
    if source is not None:
        src_file = source.text
        if "leftView" in src_file:
            return "LEFT_CAM_2K"   # (ESKİ HALİN – bilerek değiştirmedim)
        elif "rightView" in src_file:
            return "RIGHT_CAM_2K"  # (ESKİ HALİN – bilerek değiştirmedim)
        else:
            raise Exception(f"Kaynak dosyada ne leftView ne rightView var: {src_file}")
    else:
        raise Exception("XML'de <source> alanı bulunamadı.")

def load_camera_params(conf_path, cam_key):
    config = configparser.ConfigParser()
    config.read(conf_path)
    cam_params = config[cam_key]
    fx = float(cam_params['fx'])
    fy = float(cam_params['fy'])
    cx = float(cam_params['cx'])
    cy = float(cam_params['cy'])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    k1 = float(cam_params['k1'])
    k2 = float(cam_params['k2'])
    p1 = float(cam_params['p1'])
    p2 = float(cam_params['p2'])
    k3 = float(cam_params['k3'])
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    return K, dist

def load_extrinsics(json_path, cam_id):
    with open(json_path, "r") as f:
        data = json.load(f)
        extr = data[str(cam_id)]['world']
        rot_vec = np.array(extr['rotation'], dtype=np.float64)
        trans_vec = np.array(extr['translation'], dtype=np.float64)
        # (ESKİ HALİN – Euler değil Rodrigues varsayımı)
        R = cv2.Rodrigues(rot_vec)[0]
        T = trans_vec.reshape(3, 1)
        return R, T

def load_points(path):
    pts = {}
    with open(path, "r") as f:
        for line in f:
            frame, x, y = line.strip().split()
            pts[int(frame)] = [float(x), float(y)]
    return pts

def build_projection_matrix(K, R, T):
    RT = np.hstack((R, T))
    return K @ RT

def triangulate_multi_view(proj_mats, points_2d):
    A = []
    for i in range(3):
        x, y = points_2d[i]
        P = proj_mats[i]
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A)
    _, _, Vh = np.linalg.svd(A)
    X = Vh[-1]
    X = X / X[3]
    return X[:3]

def reproj_err(P, X, x):
    """Tek kamera için reprojection (piksel) hatası."""
    Xh = np.append(X, 1.0)
    u, v, w = (P @ Xh)
    u, v = u / w, v / w
    return float(np.hypot(u - x[0], v - x[1]))

if __name__ == "__main__":
    cam_ids = [38738369, 31118929, 38838483]  # Kameraların id'leri
    undistorted_files = [
        "undistorted_points/38738369_undistorted_points.txt",
        "undistorted_points/31118929_undistorted_points.txt",
        "undistorted_points/38838483_undistorted_points.txt",
    ]
    conf_files = [
        "configuration_files/SN38738369.conf",
        "configuration_files/SN31118929.conf",
        "configuration_files/SN38838483.conf",
    ]
    xml_files = [
        "xml_files/38738369.xml",
        "xml_files/31118929.xml",
        "xml_files/38838483.xml",
    ]
    json_path = "2025_03_27-1034.json"

    # Projection matrisleri (ESKİ HALİN)
    proj_mats = []
    for cam_id, conf_file, xml_file in zip(cam_ids, conf_files, xml_files):
        cam_key = get_cam_key_from_xml(xml_file)
        K, dist = load_camera_params(conf_file, cam_key)
        R, T = load_extrinsics(json_path, cam_id)
        P = build_projection_matrix(K, R, T)
        proj_mats.append(P)

    # 2D noktalar
    pts = [load_points(f) for f in undistorted_files]

    # Ortak frameler
    frames = sorted(set(pts[0].keys()) & set(pts[1].keys()) & set(pts[2].keys()))
    if not frames:
        raise Exception("Hiç ortak frame bulunamadı!")

    # Reprojection istatistikleri için akümülatör
    errs_per_cam = [[], [], []]

    # Hesapla ve dosyaya yaz (ESKİ HALİN)
    with open("triangulated_points_3cams.txt", "w") as f3d:
        for frame in frames:
            points_2d = [pts[i][frame] for i in range(3)]
            X = triangulate_multi_view(proj_mats, points_2d)
            f3d.write(f"{frame} {X[0]:.4f} {X[1]:.4f} {X[2]:.4f}\n")

            # --- YENİ: reprojection error ölç ---
            for i in range(3):
                e = reproj_err(proj_mats[i], X, points_2d[i])
                errs_per_cam[i].append(e)

    # --- YENİ: istatistik dosyası ---
    with open("reprojection_stats_v1.txt", "w") as fr:
        for i, arr in enumerate(errs_per_cam, start=1):
            if not arr:
                fr.write(f"Cam{i}: no data\n")
                continue
            a = np.array(arr, dtype=float)
            fr.write(
                f"Cam{i}: mean={a.mean():.3f} px, median={median(a):.3f} px, "
                f"max={a.max():.3f} px, N={len(a)}\n"
            )

    print("Kaydedildi: triangulated_points_3cams.txt ve reprojection_stats_v1.txt")
