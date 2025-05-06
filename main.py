import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# —— 參數區 —— #
INPUT_DIR = "images"             # 輸入圖片資料夾
OUTPUT_DIR = "outputs"           # 輸出結果資料夾
INTERMEDIATE_DIR = "intermediate"  # 中間遮罩圖存放資料夾

# 確保輸出資料夾存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

# 影像處理核心參數
BLUR_KERNEL = (5, 5)  # 高斯模糊核大小
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 形態學處理核

# 定義各顏色的 HSV 範圍
COLOR_RANGES = {
    'green': [
        (np.array([25, 50, 50]), np.array([40, 255, 255]))  # 綠色範圍
    ],
    'orange': [
        (np.array([5, 100, 100]), np.array([20, 255, 255]))  # 橙色範圍
    ],
    'white': [
        (np.array([0, 0, 220]), np.array([180, 30, 255]))   # 白色範圍
    ],
    'purple': [
        (np.array([120, 30, 30]), np.array([170, 255, 255])) # 紫色範圍
    ],
}

# 篩選圓形的最小面積與半徑範圍
MIN_AREA = 200  # 最小輪廓面積
MIN_RAD = 25    # 最小半徑
MAX_RAD = 200   # 最大半徑

# 處理每張圖片 
for filename in os.listdir(INPUT_DIR):
    # 只處理 jpg/jpeg/png
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(INPUT_DIR, filename)
    img_bgr = cv2.imread(image_path)  # 讀取 BGR 格式圖

    # 讀取失敗時跳過
    if img_bgr is None:
        print(f"❌ 無法讀取圖片：{filename}")
        continue

    # BGR 轉 RGB，並統一縮放至 800x600
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (800, 600))

    # 轉為 HSV 格式，並進行高斯模糊
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    blur = cv2.GaussianBlur(hsv, BLUR_KERNEL, 0)

    # 用 PIL 建立可繪圖物件
    output_pil = Image.fromarray(img_rgb.copy())
    draw = ImageDraw.Draw(output_pil)

    # 根據顏色範圍偵測並標記圓形
    for cname, ranges in COLOR_RANGES.items():
        # 初始化總遮罩
        mask_total = np.zeros((600, 800), dtype=np.uint8)

        # 合併可能的多組上下界遮罩
        for lower, upper in ranges:
            mask = cv2.inRange(blur, lower, upper)
            mask_total = cv2.bitwise_or(mask_total, mask)

        # 儲存原始遮罩圖
        mask_name = f"{filename}_{cname}_mask_raw.jpg"
        cv2.imwrite(os.path.join(INTERMEDIATE_DIR, mask_name), mask_total)

        # 形態學開閉運算去除雜點
        m1 = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, KERNEL)
        m2 = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, KERNEL)
        mask_proc_name = f"{filename}_{cname}_mask_proc.jpg"
        cv2.imwrite(os.path.join(INTERMEDIATE_DIR, mask_proc_name), m2)

        # 尋找輪廓
        cnts, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            # 面積太小的忽略
            if area < MIN_AREA:
                continue

            # 最小外接圓
            (x, y), r = cv2.minEnclosingCircle(cnt)
            # 半徑範圍過小或過大都忽略
            if r < MIN_RAD or r > MAX_RAD:
                continue

            # 計算輪廓的圓度指標，濾除非圓形
            per = cv2.arcLength(cnt, True)
            circ = 4 * np.pi * area / (per * per) if per > 0 else 0
            if circ < 0.6:
                continue

            # 在圖上畫圓框並標註顏色名稱
            draw.ellipse([(x - r, y - r), (x + r, y + r)], outline="red", width=2)
            draw.text((x - r, y - r - 15), cname, fill="white")

    # —— 儲存最終標註結果 —— #
    final = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGB2BGR)
    output_path = os.path.join(OUTPUT_DIR, f"annotated_{filename}")
    cv2.imwrite(output_path, final)
    print(f"✅ 已處理：{filename} → {output_path}")