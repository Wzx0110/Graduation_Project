import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

start_time = time.time()

# 參數設定
OUTPUT_DIR = "../../assets/image_alignment"  # 輸出資料夾
RATIO_THRESHOLD = 0.6         # Lowe's Ratio Test 閾值 (更嚴格，例如 0.7, 0.65, 0.6)
MIN_GOOD_MATCH_COUNT = 10      # Ratio Test 後至少需要多少個匹配點才能繼續
# Homography 估計閾值
RANSAC_REPROJ_THRESHOLD = 1.0  # RANSAC/MAGSAC 重投影誤差閾值 (像素), 嘗試 5.0, 3.0, 1.0
MIN_INLIER_COUNT = 10         # RANSAC/MAGSAC 至少需要找到的內點數

# 讀取圖片
img1_path = '1.png'  # 待對齊圖片
img2_path = '2.png'  # 參考圖片
img1_color = cv2.imread(img1_path)
img2_color = cv2.imread(img2_path)

if img1_color is None or img2_color is None:
    print(f"錯誤：無法讀取圖片。")
    exit()

img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
h1, w1 = img1_gray.shape
h2, w2 = img2_gray.shape
print("圖片讀取成功")

# 初始化 SIFT 偵測器
sift = cv2.SIFT_create()

# 偵測關鍵點與計算描述符 (RootSIFT)
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)
print(f"圖片1找到 {len(kp1)} 個關鍵點，圖片2找到 {len(kp2)} 個關鍵點。")

# RootSIFT 轉換函數


def rootsift(des):
    if des is None:
        return None
    des = np.float32(des)
    des /= (np.sum(np.abs(des), axis=1, keepdims=True) + 1e-7)
    des = np.sqrt(des)
    return des


des1 = rootsift(des1)
des2 = rootsift(des2)

# 顯示並儲存偵測到的關鍵點
img1_keypoints = cv2.drawKeypoints(
    img1_color, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_keypoints = cv2.drawKeypoints(
    img2_color, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

fname1 = os.path.join(OUTPUT_DIR, "img1_keypoints.png")
fname2 = os.path.join(OUTPUT_DIR, "img2_keypoints.png")
cv2.imwrite(fname1, img1_keypoints)
cv2.imwrite(fname2, img2_keypoints)

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img1_keypoints, cv2.COLOR_BGR2RGB))
plt.title(f'圖片 1 的 SIFT 關鍵點 ({len(kp1)}個)',
          fontproperties="Microsoft JhengHei")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img2_keypoints, cv2.COLOR_BGR2RGB))
plt.title(f'圖片 2 的 SIFT 關鍵點 ({len(kp2)}個)',
          fontproperties="Microsoft JhengHei")
plt.axis('off')
plt.suptitle('偵測到的 SIFT 關鍵點', fontproperties="Microsoft JhengHei")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 特徵匹配 (FLANN)
FLANN_INDEX_KDTREE = 1  # (KD-Tree 演算法)
# 構建的 KD-Tree 數量，越多越快，但記憶體使用量增加
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # 增加 checks 會提高精度，但降低速度
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)  # 2點最近鄰
print(f"FLANN 匹配找到 {len(matches)} 組初步匹配。")

# 篩選良好匹配 (Lowe's Ratio Test)
good_matches = []
valid_matches_count = sum(1 for pair in matches if len(pair) == 2)
if valid_matches_count > 0:
    for m, n in (pair for pair in matches if len(pair) == 2):
        if m.distance < RATIO_THRESHOLD * n.distance:
            good_matches.append(m)
    print(f"Lowe's Ratio Test 剩下 {len(good_matches)} 組良好匹配。")
else:
    print("無法進行 Ratio Test。")

# 顯示並儲存良好匹配點
if good_matches:
    img_good_matches = cv2.drawMatches(
        img1_color, kp1, img2_color, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    fname = os.path.join(OUTPUT_DIR, "good_matches.png")
    cv2.imwrite(fname, img_good_matches)
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(img_good_matches, cv2.COLOR_BGR2RGB))
    plt.title(
        f'Lowe\'s Ratio Test 後的良好匹配點 ({len(good_matches)}對)', fontproperties="Microsoft JhengHei")
    plt.axis('off')
    plt.show()
else:
    print("沒有足夠的良好匹配點可以顯示。")

# 計算 Homography 矩陣 (MAGSAC++)
if len(good_matches) >= MIN_GOOD_MATCH_COUNT:
    pts1 = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H = None  # 單應性矩陣，圖像變換
    mask = None  # 內點遮罩，指示哪些匹配點是內點

    # MAGSAC++
    H, mask = cv2.findHomography(
        pts1, pts2, cv2.USAC_MAGSAC, RANSAC_REPROJ_THRESHOLD)

    inlier_count = np.sum(mask)
    print(f"MAGSAC++找到 {inlier_count} 個內點 (Inliers)。")

    if inlier_count >= MIN_INLIER_COUNT:
        print(f"內點數量 ({inlier_count}) 達到要求 ({MIN_INLIER_COUNT})。")
        matchesMask = mask.ravel().tolist()

        # 顯示並儲存內點匹配
        img_magsac_inliers = cv2.drawMatches(img1_color, kp1, img2_color, kp2, good_matches,
                                             None, matchesMask=matchesMask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        fname = os.path.join(OUTPUT_DIR, f"MAGSAC++_inlier_matches.png")
        cv2.imwrite(fname, img_magsac_inliers)

        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(img_magsac_inliers, cv2.COLOR_BGR2RGB))
        plt.title(f'MAGSAC++ 內點匹配 ({inlier_count}/{len(good_matches)}對)',
                  fontproperties="Microsoft JhengHei")
        plt.axis('off')
        plt.show()

        # 顯示並儲存圖像對齊
        aligned_img1 = cv2.warpPerspective(
            img1_color, H, (w2, h2), flags=cv2.INTER_CUBIC)

        fname = os.path.join(OUTPUT_DIR, "aligned_image1.png")
        cv2.imwrite(fname, aligned_img1)

        blended_image = cv2.addWeighted(
            img2_color, 0.5, aligned_img1, 0.5, 0.0)

        fname = os.path.join(OUTPUT_DIR, "blended_result.png")
        cv2.imwrite(fname, blended_image)

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB))
        plt.title('圖片 2 (參考)', fontproperties="Microsoft JhengHei")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(aligned_img1, cv2.COLOR_BGR2RGB))
        plt.title('圖片 1 (對齊到圖片 2)', fontproperties="Microsoft JhengHei")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
        plt.title('疊加結果', fontproperties="Microsoft JhengHei")
        plt.axis('off')
        plt.suptitle(f'RootSIFT + MAGSAC++ 圖像對齊結果',
                     fontproperties="Microsoft JhengHei")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    else:
        print(f"警告：找到的內點數量 ({inlier_count}) 不足 {MIN_INLIER_COUNT}。")
        print(
            "建議嘗試調整閾值 (RATIO_THRESHOLD, RANSAC_REPROJ_THRESHOLD, MIN_INLIER_COUNT) 或檢查圖片。")

else:
    print(
        f"錯誤：找不到足夠的良好匹配點 ({len(good_matches)} / {MIN_GOOD_MATCH_COUNT}) 來計算 Homography。")
    print("試放寬 RATIO_THRESHOLD (例如 0.7, 0.75) 或檢查圖片本身。")

print("--- 影像對齊處理完成 ---")
