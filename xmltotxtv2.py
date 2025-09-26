from lxml import etree
from pathlib import Path

# === KULLANICI AYARI ===
FRAME_RANGES = [
    (1162, 1202),
    (1439, 1481),
    (1729, 1770),
    (2222, 2273),
    (2499, 2544),
]

def get_labels_from_xml(xml_path):
    root = etree.parse(xml_path).getroot()
    return [n.text for n in root.findall('.//label/name')]

def read_ball_points(xml_path, label_name):
    root = etree.parse(xml_path).getroot()
    ball_points = []
    track = root.find(f'.//track[@label="{label_name}"]')
    if track is None:
        print(f"{label_name} etiketi yok: {xml_path}")
        return ball_points
    for pt in track.findall("points"):
        frame = int(pt.get("frame"))
        points_str = pt.get("points")
        if points_str:
            x_str, y_str = points_str.split(",")
            ball_points.append((frame, float(x_str), float(y_str)))
    return ball_points

def filter_points_by_ranges(points, ranges):
    keep = []
    for f, x, y in points:
        for a, b in ranges:
            if a <= f <= b:
                keep.append((f, x, y))
                break
    return keep

def write_points_to_txt(points, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for frame, x, y in points:
            f.write(f"{frame} {x:.2f} {y:.2f}\n")

def process_xml_to_txt(xml_path, out_path, frame_ranges=FRAME_RANGES):
    labels = get_labels_from_xml(xml_path)
    if not labels:
        print(f"Label yok: {xml_path}")
        return
    label = labels[0]
    pts = read_ball_points(xml_path, label)
    filt = filter_points_by_ranges(pts, frame_ranges)
    write_points_to_txt(filt, out_path)
    print(f"[OK] {out_path} <- {len(filt)} nokta")

if __name__ == "__main__":
    datasets = [
        ("xml_files/38738369.xml", "filtered_xml_to_txt_points_v2/38738369_filtered_points_v2.txt"),
        ("xml_files/31118929.xml", "filtered_xml_to_txt_points_v2/31118929_filtered_points_v2.txt"),
        ("xml_files/38838483.xml", "filtered_xml_to_txt_points_v2/38838483_filtered_points_v2.txt"),
    ]
    for xml_path, out_path in datasets:
        process_xml_to_txt(xml_path, out_path)
