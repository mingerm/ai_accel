import cv2
import pytesseract
import matplotlib.pyplot as plt
import re

video_path = "number.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("영상 파일을 열 수 없습니다:", video_path)
    exit()

frame_num = 0
history = []
last_num = None

plt.ion()
fig = plt.figure(figsize=(12, 5))

while True:
    ret, frame = cap.read()

    if not ret:
        print("영상 종료")
        break

    frame_num += 1

    # 5프레임마다 OCR
    if frame_num % 5 != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    config = (
        "--psm 10 "
        "-c tessedit_char_whitelist=0123456789"
    )

    text = pytesseract.image_to_string(
        thresh,
        config=config
    )

    nums = re.findall(r"\d+", text)

    if nums:
        current = nums[0]

        if current != last_num:
            history.append(current)
            last_num = current

    plt.clf()

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(f"Frame {frame_num}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(thresh, cmap="gray")
    plt.title("Threshold")
    plt.axis("off")

    plt.suptitle(
        f"Current: {nums} | History: {' -> '.join(history)}"
    )

    plt.draw()
    plt.pause(0.001)

    print("현재 인식:", nums)
    print("인식 히스토리:", " → ".join(history))
    print("-" * 40)

cap.release()

plt.ioff()
plt.show()