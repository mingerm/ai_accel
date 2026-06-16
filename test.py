import cv2
import shutil
from pathlib import Path

BASE = Path.cwd() 
SELECTED = BASE / "8"

OUT_IMG = BASE / "yolo_auto" / "images"
OUT_LBL = BASE / "yolo_auto" / "labels"

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

# digits = ["1", "3", "5", "6", "8"]
digits = ["8"]
exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]

def make_bbox(img):
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 칠판/종이 배경은 밝고 숫자는 어두운 경우가 많으므로 invert threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7
    )

    # 작은 잡음 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh

        # 너무 작은 잡음 제거
        if area < (w * h) * 0.001:
            continue

        # 너무 큰 배경 오검출 제거
        if area > (w * h) * 0.8:
            continue

        boxes.append((x, y, bw, bh))

    if not boxes:
        return None

    # 숫자 획이 여러 contour로 나뉠 수 있으니 전체 contour를 하나의 bbox로 합침
    x1 = min(x for x, y, bw, bh in boxes)
    y1 = min(y for x, y, bw, bh in boxes)
    x2 = max(x + bw for x, y, bw, bh in boxes)
    y2 = max(y + bh for x, y, bw, bh in boxes)

    # 여백 조금 추가
    pad_x = int((x2 - x1) * 0.25)
    pad_y = int((y2 - y1) * 0.15)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    bw = x2 - x1
    bh = y2 - y1

    # YOLO 형식: class center_x center_y width height
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    nw = bw / w
    nh = bh / h

    return cx, cy, nw, nh

count = 0
failed = []

for digit in digits:
    src_dir = SELECTED 

    files = []
    for ext in exts:
        files.extend(src_dir.glob(ext))

    for src in sorted(files):
        img = cv2.imread(str(src))
        if img is None:
            failed.append(str(src))
            continue

        bbox = make_bbox(img)
        if bbox is None:
            failed.append(str(src))
            continue

        new_name = f"{digit}_{src.stem}{src.suffix.lower()}"
        out_img_path = OUT_IMG / new_name
        out_lbl_path = OUT_LBL / f"{Path(new_name).stem}.txt"

        shutil.copy2(src, out_img_path)

        cx, cy, nw, nh = bbox
        with open(out_lbl_path, "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        count += 1

print(f"Created YOLO labels: {count}")
print(f"Failed images: {len(failed)}")

if failed:
    print("Failed list:")
    for f in failed[:30]:
        print(f)