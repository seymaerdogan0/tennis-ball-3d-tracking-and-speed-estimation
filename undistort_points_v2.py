import numpy as np
import cv2
import configparser
import xml.etree.ElementTree as ET
from pathlib import Path

def read_points(txt_path):
    pts = []
    with open(txt_path, "r") as f:
        for line in f:
            frame, x, y = line.strip().split()
            pts.append({"frame": int(frame), "x": float(x), "y": float(y)})
    return pts

def load_camera_params(conf_path, cam_key):
    cfg = configparser.ConfigParser()
    cfg.read(conf_path)
    cam = cfg[cam_key]
    fx, fy = float(cam['fx']), float(cam['fy'])
    cx, cy = float(cam['cx']), float(cam['cy'])
    k1, k2 = float(cam['k1']), float(cam['k2'])
    p1, p2 = float(cam['p1']), float(cam['p2'])
    k3 = float(cam['k3'])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    return K, D

def undistort_points(points, K, D):
    pts = np.array([[p['x'], p['y']] for p in points], dtype=np.float32).reshape(-1,1,2)
    # P=K vererek çıkışı tekrar piksel uzayında isteriz
    und = cv2.undistortPoints(pts, K, D, P=K)
    out = []
    for i, p in enumerate(points):
        u, v = und[i,0]
        out.append({"frame": p['frame'], "x": float(u), "y": float(v)})
    return out

def save_points(points, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in points:
            f.write(f"{p['frame']} {p['x']:.2f} {p['y']:.2f}\n")

def get_cam_key_from_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    src  = root.findtext(".//source")
    W = int(root.findtext(".//original_size/width"))
    H = int(root.findtext(".//original_size/height"))
    side = "LEFT" if ("leftView" in src) else "RIGHT"
    if   (W,H)==(1280,720): res="HD"
    elif (W,H)==(1920,1080): res="FHD"
    else: res="2K"
    return f"{side}_CAM_{res}"

if __name__ == "__main__":
    datasets = [
        {
            "cam_id": "38738369",
            "txt_path": "filtered_xml_to_txt_points_v2/38738369_filtered_points_v2.txt",
            "conf_path": "configuration_files/SN38738369.conf",
            "xml_path": "xml_files/38738369.xml",
            "out_path": "undistorted_points_v2/38738369_undistorted_points_v2.txt",
        },
        {
            "cam_id": "31118929",
            "txt_path": "filtered_xml_to_txt_points_v2/31118929_filtered_points_v2.txt",
            "conf_path": "configuration_files/SN31118929.conf",
            "xml_path": "xml_files/31118929.xml",
            "out_path": "undistorted_points_v2/31118929_undistorted_points_v2.txt",
        },
        {
            "cam_id": "38838483",
            "txt_path": "filtered_xml_to_txt_points_v2/38838483_filtered_points_v2.txt",
            "conf_path": "configuration_files/SN38838483.conf",
            "xml_path": "xml_files/38838483.xml",
            "out_path": "undistorted_points_v2/38838483_undistorted_points_v2.txt",
        },
    ]

    for ds in datasets:
        cam_key = get_cam_key_from_xml(ds["xml_path"])
        pts = read_points(ds["txt_path"])
        K, D = load_camera_params(ds["conf_path"], cam_key)
        und = undistort_points(pts, K, D)
        save_points(und, ds["out_path"])
        print(f"[OK] {ds['out_path']}  (Kamera: {cam_key})")
