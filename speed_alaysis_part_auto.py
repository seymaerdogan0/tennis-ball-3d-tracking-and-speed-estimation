import numpy as np
import matplotlib.pyplot as plt

def read_3d_points(path):
    frames, xs, ys, zs = [], [], [], []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            frames.append(int(parts[0]))
            xs.append(float(parts[1]))
            ys.append(float(parts[2]))
            zs.append(float(parts[3]))
    return np.array(frames), np.array(xs), np.array(ys), np.array(zs)

def find_intervals(frames, gap_threshold=10):
    intervals = []
    start = frames[0]
    for i in range(1, len(frames)):
        if frames[i] - frames[i-1] > gap_threshold:
            end = frames[i-1]
            intervals.append((start, end))
            start = frames[i]
    intervals.append((start, frames[-1]))
    return intervals

def calculate_speed_piecewise(frames, xs, ys, zs, intervals, fps=30):
    speed_segments = []
    for giris, cikis in intervals:
        mask = (frames >= giris) & (frames <= cikis)
        seg_frames = frames[mask]
        seg_xs = xs[mask]
        seg_ys = ys[mask]
        seg_zs = zs[mask]
        coords = np.vstack([seg_xs, seg_ys, seg_zs]).T
        speeds, times = [], []
        for i in range(1, len(seg_frames)):
            dt = (seg_frames[i] - seg_frames[i-1]) / fps
            dist = np.linalg.norm(coords[i] - coords[i-1])
            v = dist / dt if dt > 0 else 0
            speeds.append(v)
            times.append(seg_frames[i] / fps)
        if len(times) > 0:
            speed_segments.append((times, speeds))
    return speed_segments

def plot_speed_segments(speed_segments, intervals):
    plt.figure(figsize=(18,8))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'gray', 'pink']
    all_speeds = np.concatenate([s for t, s in speed_segments if len(s) > 0])
    max_hiz = np.max(all_speeds)
    plt.ylim(0, max_hiz * 1.2)
    for i, (times, speeds) in enumerate(speed_segments):
        c = colors[i % len(colors)]
        plt.plot(times, speeds, label=f"Parça {i+1} ({intervals[i][0]}-{intervals[i][1]})", color=c, linewidth=1.5)
        plt.scatter(times, speeds, color=c, s=10) # marker boyutu küçük
    plt.xlabel("Zaman (sn)", fontsize=14)
    plt.ylabel("Hız (m/s)", fontsize=14)
    plt.title("Topun Parça Parça Hız Grafiği", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    filename = "triangulated_points_3cams.txt"
    fps = 30
    frames, xs, ys, zs = read_3d_points(filename)
    intervals = find_intervals(frames, gap_threshold=10)
    print("Bulunan aralıklar (giriş/çıkış):", intervals)
    speed_segments = calculate_speed_piecewise(frames, xs, ys, zs, intervals, fps)
    plot_speed_segments(speed_segments, intervals)