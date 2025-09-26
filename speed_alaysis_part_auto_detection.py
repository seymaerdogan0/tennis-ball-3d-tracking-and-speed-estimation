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
            speed_segments.append((np.array(times), np.array(speeds)))
    return speed_segments

def detect_physical_events_sequential(times, speeds, threshold=6, window=5):
    speed_diff = np.diff(speeds)
    # Ani hızlanma noktalarını bul (eşik üstü farklar)
    event_idxs = np.where(speed_diff > threshold)[0] + 1
    # Sıralı olarak olay noktalarını bul
    minima_idxs = []
    for idx in event_idxs:
        search_start = max(0, idx-window)
        search_end = idx
        if search_end > search_start:  # en az bir eleman olsun
            local_min_idx = search_start + np.argmin(speeds[search_start:search_end])
            # Aynı minimumu birden fazla eklememek için kontrol
            if len(minima_idxs) == 0 or local_min_idx != minima_idxs[-1]:
                minima_idxs.append(local_min_idx)
    # Her zaman en az bir olay işaretle
    ground_idx = minima_idxs[0] if len(minima_idxs) > 0 else np.argmin(speeds)
    ground_time = times[ground_idx]
    ground_speed = speeds[ground_idx]
    # İkinci olay varsa raket vuruşu olarak işaretle
    if len(minima_idxs) > 1:
        racket_idx = minima_idxs[1]
        racket_time = times[racket_idx]
        racket_speed = speeds[racket_idx]
    elif len(minima_idxs) == 1:
        # Sadece bir ani hızlanma varsa, ikinci minimumu grafik sonunda ara
        # En büyük hız noktası öncesi bir minimum var mı?
        max_idx = np.argmax(speeds)
        search_start = max(0, max_idx-window)
        search_end = max_idx
        if search_end > search_start:
            racket_idx = search_start + np.argmin(speeds[search_start:search_end])
            if racket_idx == ground_idx:
                racket_idx = None
        else:
            racket_idx = None
        if racket_idx is not None:
            racket_time = times[racket_idx]
            racket_speed = speeds[racket_idx]
        else:
            racket_time, racket_speed = None, None
    else:
        racket_time, racket_speed = None, None
    return ground_time, ground_speed, racket_time, racket_speed

def plot_speed_segments_with_events(speed_segments, intervals):
    plt.figure(figsize=(18,8))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'gray', 'pink']
    all_speeds = np.concatenate([s for t, s in speed_segments if len(s) > 0])
    max_hiz = np.max(all_speeds)
    plt.ylim(0, max_hiz * 1.2)
    for i, (times, speeds) in enumerate(speed_segments):
        c = colors[i % len(colors)]
        plt.plot(times, speeds, label=f"Parça {i+1} ({intervals[i][0]}-{intervals[i][1]})", color=c, linewidth=1.5)
        ground_time, ground_speed, racket_time, racket_speed = detect_physical_events_sequential(times, speeds, threshold=6, window=5)
        # Yerde sekme
        plt.scatter(ground_time, ground_speed, color='deepskyblue', s=120, marker='o', label='Yerde Sekme' if i==0 else "")
        # Raket vuruşu
        if racket_time is not None and racket_time != ground_time:
            plt.scatter(racket_time, racket_speed, color='green', s=120, marker='s', label='Raket Vuruşu' if i==0 else "")
    plt.xlabel("Zaman (sn)", fontsize=14)
    plt.ylabel("Hız (m/s)", fontsize=14)
    plt.title("Topun Parça Parça Hız Grafiği ve Fiziksel Olaylar", fontsize=16)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    filename = "triangulated_points_3cams_v5b.txt"
    fps = 60.0
    frames, xs, ys, zs = read_3d_points(filename)
    intervals = find_intervals(frames, gap_threshold=10)
    print("Bulunan aralıklar (giriş/çıkış):", intervals)
    speed_segments = calculate_speed_piecewise(frames, xs, ys, zs, intervals, fps)
    plot_speed_segments_with_events(speed_segments, intervals)