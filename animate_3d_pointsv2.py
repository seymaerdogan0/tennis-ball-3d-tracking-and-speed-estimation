# animate_3d_points_segments_v2.py
# Kırık (ardışık olmayan) frame aralıklarını ayrı ayrı oynatır (iz her segmentte sıfırlanır)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --------- AYARLAR ---------
IN_FILE = "triangulated_points_3cams_v5b.txt"
FPS = 60.0                 # REALTIME=True ise gerçek-zaman hızı
REALTIME = True
TAIL_LEN = 40              # segment içi kuyruk uzunluğu

USE_ZUP_AND_BOUNCE = False # Otomatik hizalama (Z up + sekme orijin + yön hizası)
FORCE_VIEW = True
VIEW_AZIM, VIEW_ELEV = -135, 25

# MANUEL remap -> HER ZAMAN EN SON UYGULANIR
MANUAL_AXIS_ORDER = ('x','y','z')    # örn: ('y','z','x')
MANUAL_SIGN       = (+1, +1, -1)     # örn: (-1, +1, +1)

# --------- YARDIMCI ---------
def read_3d(path):
    fr, X = [], []
    with open(path, "r", encoding="utf-8") as f:
        for L in f:
            p = L.strip().split()
            if len(p) < 4: continue
            fr.append(int(p[0]))
            X.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(fr), np.array(X, float)

def split_contiguous(frames):
    """Ardışık olmayan yerlerden kesitler oluştur."""
    cuts = [0]
    for i in range(1, len(frames)):
        if frames[i] != frames[i-1] + 1:
            cuts.append(i)
    cuts.append(len(frames))
    return [(cuts[i], cuts[i+1]) for i in range(len(cuts)-1)]  # (start_idx, end_idx)

def quad_fit_acc(t,y):
    A = np.vstack([t**2, t, np.ones_like(t)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return 2.0*c[0]

def estimate_vertical_axis(frames, X):
    segs = split_contiguous(frames)
    accs = []
    for s,e in segs:
        if e-s < 7: continue
        t = (frames[s:e] - frames[s]) / FPS
        accs.append([quad_fit_acc(t, X[s:e,j]) for j in range(3)])
    if not accs: return 2, -9.81
    accs = np.array(accs); mean = accs.mean(axis=0)
    axis = int(np.argmax(np.abs(mean)))
    return axis, mean[axis]

def reorder_axes(X, order=('x','y','z')):
    idx = {'x':0,'y':1,'z':2}
    return X[:, [idx[order[0]], idx[order[1]], idx[order[2]]]]

def rotate_xy_to_direction(X, idx_start, idx_end):
    """XY düzleminde ilk hareket yönünü +X'e çevir."""
    XY = X[:, :2].copy()
    i0 = max(0, idx_start); i1 = max(i0+5, min(idx_end, idx_start+10))
    v = XY[i1] - XY[i0]
    if np.linalg.norm(v) < 1e-6: return X
    phi = np.arctan2(v[1], v[0])
    R = np.array([[np.cos(-phi), -np.sin(-phi)],
                  [np.sin(-phi),  np.cos(-phi)]])
    XYr = (R @ XY.T).T
    Xr = np.column_stack([XYr, X[:,2]])
    if np.mean(np.diff(Xr[max(0, idx_start-5):idx_end, 0])) < 0:
        Xr[:,0] *= -1.0
    return Xr

# --------- VERİ + REMAP ---------
frames, X = read_3d(IN_FILE)

if USE_ZUP_AND_BOUNCE:
    # Z up + g negatif
    axis, gmean = estimate_vertical_axis(frames, X)
    order = ['x','y','z']; order[axis], order[2] = order[2], order[axis]
    X = reorder_axes(X, tuple(order))
    if gmean > 0: X[:,2] *= -1.0
    # en uzun segmentin (muhtemel sekme) orijine alınması
    segs = split_contiguous(frames)
    s,e = max(segs, key=lambda se: se[1]-se[0])
    b = s + int(np.argmin(X[s:e,2]))
    X = X - X[b]
    X = rotate_xy_to_direction(X, max(s, b-10), b)

# Manuel remap (her zaman en son!)
X = reorder_axes(X, MANUAL_AXIS_ORDER)
X *= np.array(MANUAL_SIGN)[None, :]
print(f"[INFO] Manual remap: order={MANUAL_AXIS_ORDER}, sign={MANUAL_SIGN}")

# --------- SEGMENT ZAMAN ÇİZELGESİ ---------
# Her animasyon adımı (seg_idx, local_i) olarak tanımlanır
segments = split_contiguous(frames)
seg_arrays = [X[s:e] for (s,e) in segments]
timeline = [(k, i) for k, arr in enumerate(seg_arrays) for i in range(len(arr))]

# --------- ANİMASYON ---------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# eşit ölçekli kutu
xmin, xmax = X[:,0].min(), X[:,0].max()
ymin, ymax = X[:,1].min(), X[:,1].max()
zmin, zmax = X[:,2].min(), X[:,2].max()
cx, cy, cz = (xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2
rng = max(xmax-xmin, ymax-ymin, zmax-zmin)
ax.set_xlim(cx - rng/2, cx + rng/2)
ax.set_ylim(cy - rng/2, cy + rng/2)
ax.set_zlim(cz - rng/2, cz + rng/2)

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
plt.title(f"3D Hareketli Nokta: {IN_FILE}")
if FORCE_VIEW: ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

(line,) = ax.plot([], [], [], lw=2, color='tab:red')
(dot,)  = ax.plot([], [], [], 'o', ms=6, color='tab:red')
last_seg = {'idx': -1}

def init():
    line.set_data([], []); line.set_3d_properties([])
    dot.set_data([], []);  dot.set_3d_properties([])
    return line, dot

def update(step):
    seg_idx, i = timeline[step]
    arr = seg_arrays[seg_idx]

    # segment değiştiyse izi sıfırla
    if seg_idx != last_seg['idx']:
        line.set_data([], []); line.set_3d_properties([])
        last_seg['idx'] = seg_idx

    i0 = max(0, i - TAIL_LEN)
    line.set_data(arr[i0:i+1,0], arr[i0:i+1,1])
    line.set_3d_properties(arr[i0:i+1,2])
    dot.set_data([arr[i,0]], [arr[i,1]])
    dot.set_3d_properties([arr[i,2]])

    ax.set_title(f"3D Hareketli Nokta: {IN_FILE}  |  segment {seg_idx+1}/{len(seg_arrays)}  |  idx {i+1}/{len(arr)}")
    return line, dot

interval_ms = (1000.0 / FPS) if REALTIME else 40.0
ani = FuncAnimation(fig, update, init_func=init,
                    frames=len(timeline), interval=interval_ms, blit=True)

plt.tight_layout()
plt.show()

# Kaydetmek istersen:
# ani.save("animation_segments_v2.mp4", writer="ffmpeg", fps=FPS)
