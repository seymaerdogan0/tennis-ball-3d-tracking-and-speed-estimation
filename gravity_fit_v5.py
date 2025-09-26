import argparse
import numpy as np

# Boş uçuş segmentleri (temas kareleri kırpılmış)
SEGMENTS = [
    (1459, 1468),
    (1749, 1757),
    (2250, 2260),
    (2521, 2531),
]

def read_3d(path):
    fr, X = [], []
    with open(path, "r", encoding="utf-8") as f:
        for L in f:
            s = L.strip().split()
            if len(s) < 4: continue
            fr.append(int(s[0]))
            X.append([float(s[1]), float(s[2]), float(s[3])])
    return np.array(fr), np.array(X, float)

def robust_quad_fit(t, z):
    # z = a t^2 + b t + c  -> g = 2a
    A = np.vstack([t**2, t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    # 2 tur robustlaştırma (MAD)
    for _ in range(2):
        res = z - A @ coef
        mad = np.median(np.abs(res)) + 1e-9
        thr = 3.0 * 1.4826 * mad
        m = np.abs(res) <= thr
        if m.sum() < 6: break
        coef, *_ = np.linalg.lstsq(A[m], z[m], rcond=None)
    return coef  # (a,b,c)

def choose_vertical_axis(X, fps):
    # |mean(acc)| en büyük ekseni dikey say
    dt = 1.0 / fps
    def acc(sig):
        return (sig[2:] - 2*sig[1:-1] + sig[:-2]) / (dt*dt)
    means = [abs(acc(X[:,k]).mean()) for k in range(3)]
    return int(np.argmax(means))  # 0:X,1:Y,2:Z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="triangulated_points_3cams_v5.txt")
    ap.add_argument("--outfile", default="triangulated_points_3cams_v5_scaled.txt")
    ap.add_argument("--fps", type=float, default=24.0)
    args = ap.parse_args()

    frames, X = read_3d(args.infile)
    assert len(frames) == len(X) and len(X) > 10

    # Dikey ekseni seç
    k = choose_vertical_axis(X, args.fps)

    # Segmentlerden g ölç
    gs = []
    for (fs, fe) in SEGMENTS:
        i0 = np.where(frames == fs)[0]
        i1 = np.where(frames == fe)[0]
        if len(i0)==0 or len(i1)==0: continue
        i0, i1 = int(i0[0]), int(i1[0])
        if i1 - i0 + 1 < 8: continue
        t = (frames[i0:i1+1] - frames[i0]) / args.fps
        z = X[i0:i1+1, k]
        a, b, c = robust_quad_fit(t, z)
        g = 2.0 * a
        gs.append(g)

    if not gs:
        print("Uyarı: Segmentlerden g ölçülemedi.")
        return

    gs = np.array(gs, float)
    g_med = float(np.median(gs))
    s = 9.81 / max(1e-9, abs(g_med))  # ölçek katsayısı

    print(f"Dikey eksen: {['X','Y','Z'][k]}")
    print("Segment g'leri:", ["%.3f" % g for g in gs])
    print(f"g_median = {g_med:.4f}  ->  scale s = 9.81/|g_med| = {s:.4f}")

    # Ölçek uygula
    Xs = X * s

    # Doğrulama: yeniden g hesapla
    gs2 = []
    for (fs, fe) in SEGMENTS:
        i0 = np.where(frames == fs)[0]
        i1 = np.where(frames == fe)[0]
        if len(i0)==0 or len(i1)==0: continue
        i0, i1 = int(i0[0]), int(i1[0])
        if i1 - i0 + 1 < 8: continue
        t = (frames[i0:i1+1] - frames[i0]) / args.fps
        z = Xs[i0:i1+1, k]
        a, b, c = robust_quad_fit(t, z)
        gs2.append(2.0 * a)

    print("Ölçek SONRASI segment g'leri:", ["%.3f" % g for g in gs2])
    print(f"g_median_after ≈ {np.median(gs2):.4f} m/s^2")

    # Yaz
    with open(args.outfile, "w", encoding="utf-8") as f:
        for fr, (x,y,z) in zip(frames, Xs):
            f.write(f"{fr} {x:.6f} {y:.6f} {z:.6f}\n")
    print("Yazıldı ->", args.outfile)

if __name__ == "__main__":
    main()
