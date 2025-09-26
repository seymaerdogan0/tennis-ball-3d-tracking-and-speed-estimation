# visualize_3d_like_animate_v2.py
# Animasyondaki hizayı uygular: Z up tespiti -> sekme orijine taşı -> yönü +X'e çevir
# NOT: Kırmızı sekme noktası gösterilmez.

import numpy as np
import matplotlib.pyplot as plt

IN_FILE   = "triangulated_points_3cams_v5b.txt"

FORCE_VIEW = True
VIEW_AZIM, VIEW_ELEV = -96, 26      # animasyona benzer açı

# En sonda uygulanacak küçük manuel remap (Z'yi aşağı doğru göstermek için -1)
MANUAL_AXIS_ORDER = ('x','y','z')
MANUAL_SIGN       = (+1, +1, -1)

FPS = 60.0  # sadece Z-up tahmini için kullanılıyor

# ---------- yardımcılar ----------
def read_3d(path):
    fr, X = [], []
    for L in open(path, encoding="utf-8"):
        p = L.strip().split()
        if len(p) < 4: continue
        fr.append(int(p[0])); X.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(fr), np.array(X, float)

def split_contiguous(frames):
    cuts=[0]
    for i in range(1,len(frames)):
        if frames[i] != frames[i-1] + 1:
            cuts.append(i)
    cuts.append(len(frames))
    return [(cuts[i], cuts[i+1]) for i in range(len(cuts)-1)]

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
    accs = np.array(accs); mean = accs.mean(axis=0)
    axis = int(np.argmax(np.abs(mean)))
    return axis, mean[axis]

def reorder_axes(X, order=('x','y','z')):
    idx={'x':0,'y':1,'z':2}
    return X[:, [idx[order[0]], idx[order[1]], idx[order[2]]]]

def rotate_xy_to_direction(X, i0, i1):
    XY = X[:, :2].copy()
    v = XY[i1] - XY[i0]
    if np.linalg.norm(v) < 1e-6: return X
    phi = np.arctan2(v[1], v[0])
    c, s = np.cos(-phi), np.sin(-phi)
    R = np.array([[c,-s],[s,c]])
    XYr = (R @ XY.T).T
    Xr = np.column_stack([XYr, X[:,2]])
    # akış +X yönüne baksın
    if np.mean(np.diff(Xr[i0:i1+1,0])) < 0:
        Xr[:,0] *= -1.0
    return Xr

# ---------- akış ----------
frames, X = read_3d(IN_FILE)

# 1) Z-up eksenini bul, Z'ye taşı
axis, gmean = estimate_vertical_axis(frames, X)
order = ['x','y','z']; order[axis], order[2] = order[2], order[axis]
X = reorder_axes(X, tuple(order))
if gmean > 0:   # yerçekimi aşağı olmalı
    X[:,2] *= -1.0

# 2) en uzun kesitte sekme noktasını orijine al ve yönü +X'e hizala
segs = split_contiguous(frames)
s,e = max(segs, key=lambda se: se[1]-se[0])
b = s + int(np.argmin(X[s:e,2]))          # sekme ~ Z-min
X = X - X[b]
i0 = max(s, b-8); i1 = max(s, b-2)        # sekme öncesi kısa aralık
X = rotate_xy_to_direction(X, i0, i1)

# 3) en sonda küçük manuel remap (görünüm düzeltmesi)
X = reorder_axes(X, MANUAL_AXIS_ORDER)
X *= np.array(MANUAL_SIGN)[None, :]

# ---------- çizim ----------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X[:,0], X[:,1], X[:,2], c=frames, cmap='viridis', s=18)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
plt.colorbar(sc, ax=ax, label="Frame no")
plt.title(f"3D Noktalar: {IN_FILE} (animate görünümü)")

if FORCE_VIEW:
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

plt.tight_layout(); plt.show()
