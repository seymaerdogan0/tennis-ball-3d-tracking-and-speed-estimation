import cv2, numpy as np

p = r"videos\SVO_SN38738369.leftView.mp4"
cap = cv2.VideoCapture(p)
print("Reported FPS:", cap.get(cv2.CAP_PROP_FPS))
times = []
for _ in range(300):  # ilk 300 frame
    ok = cap.grab()
    if not ok: break
    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    times.append(t)
cap.release()

dt = np.diff(times)
print("mean dt:", dt.mean(), "std dt:", dt.std(), "≈ FPS:", 1.0/dt.mean())
