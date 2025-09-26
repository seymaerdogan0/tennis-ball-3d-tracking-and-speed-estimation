# plot_speed_bounce_v2.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

IN_FILE = "triangulated_points_3cams_v5.txt"
FPS = 60.0                      # gerçek fps
SG_WIN = 9                      # tek sayı, 7-11 arası deneyebilirsin
SG_POLY = 2
BOTTOM_FRAC = 0.20              # Z alt %20'lik alanında hız min fallback
MIN_SEG_LEN = 8

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
    cuts=[0]
    for i in range(1,len(frames)):
        if frames[i] != frames[i-1] + 1:
            cuts.append(i)
    cuts.append(len(frames))
    return [(cuts[i], cuts[i+1]) for i in range(len(cuts)-1)]

def smooth_xyz(X):
    Xs = X.copy()
    # kısa segmentlerde pencereyi küçült
    win = SG_WIN if len(X) >= SG_WIN else (len(X)//2*2+1)
    if win >= 5:
        for k in range(3):
            Xs[:,k] = savgol_filter(X[:,k], window_length=win, polyorder=min(SG_POLY, win-1))
    return Xs

def deriv_central(Y, dt):
    # merkezi fark; uçlarda ileri/geri fark
    V = np.zeros_like(Y)
    V[1:-1] = (Y[2:] - Y[:-2])/(2*dt)
    V[0]    = (Y[1]  - Y[0]) / dt
    V[-1]   = (Y[-1] - Y[-2])/ dt
    return V

def pick_bounce_idx(fr, Xs):
    """
    1) Z minimumu aday
    2) v_z negatif->pozitif geçişi var mı kontrol
    3) emin değilsek: Z alt %20 bölgesinde |v| minimumunu seç
    """
    t = (fr - fr[0]) / FPS
    dt = np.median(np.diff(t)) if len(t) > 1 else 1.0/FPS

    Z = Xs[:,2]
    V = np.column_stack([deriv_central(Xs[:,0],dt),
                         deriv_central(Xs[:,1],dt),
                         deriv_central(Xs[:,2],dt)])
    speed = np.linalg.norm(V, axis=1)

    i_zmin = int(np.argmin(Z))
    vz = V[:,2]

    # v_z işaret değişimi kontrolü (neg->pozitif) yakın çevrede
    i0 = max(0, i_zmin-2); i1 = min(len(fr)-1, i_zmin+2)
    neg_before = np.any(vz[max(0, i0-3):i_zmin] < 0)
    pos_after  = np.any(vz[i_zmin:min(len(fr), i1+3)] > 0)
    if neg_before and pos_after:
        return i_zmin, speed, V

    # fallback: Z alt %20 içinde hız min
    zmin, zmax = Z.min(), Z.max()
    thresh = zmin + (zmax - zmin)*BOTTOM_FRAC
    mask = Z <= thresh if zmax > zmin else np.ones_like(Z, dtype=bool)
    if np.any(mask):
        cand = np.argmin(np.where(mask, speed, np.inf))
        return int(cand), speed, V
    else:
        return i_zmin, speed, V

def main():
    frames, X = read_3d(IN_FILE)
    segs = split_contiguous(frames)

    # hız grafiği: her parçayı yan yana göster
    n = len(segs)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), sharey=True)
    if n == 1: axes = [axes]

    for ax, (s,e) in zip(axes, segs):
        if e-s < MIN_SEG_LEN:
            ax.set_axis_off(); continue

        fr_seg = frames[s:e]
        Xs = smooth_xyz(X[s:e])

        idx, speed, V = pick_bounce_idx(fr_seg, Xs)
        t = (fr_seg - fr_seg[0]) / FPS

        # çiz
        ax.plot(t, speed, lw=1.5)
        ax.scatter([t[idx]], [speed[idx]], c='deepskyblue', s=60, zorder=3, label="Yerde sekme (tahmin)")

        # bilgilendirici başlık (frame aralığı)
        ax.set_title(f"Parça {fr_seg[0]}–{fr_seg[-1]}")
        ax.set_xlabel("zaman (s)")

        # ayrıca istersen dikey hız da gör:
        # ax2 = ax.twinx(); ax2.plot(t, V[:,2], 'r--', alpha=0.4); ax2.set_ylabel("v_z (m/s)", color='r')

    axes[0].set_ylabel("Hız |v| (m/s)")
    axes[0].legend(loc="upper left")
    plt.suptitle("Hız profili ve tahmini sekme anları (Z-min + v_z geçişi)", y=0.98)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
