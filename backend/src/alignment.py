import cv2
import numpy as np
import time
import os

# 參數設定
OUTPUT_DIR = "../../assets/image_alignment"  # 輸出資料夾
RATIO_THRESHOLD = 0.6                  # Lowe 比率測試的閾值
MIN_GOOD_MATCH_COUNT = 10              # 至少需要多少個良好匹配點才能估計 Homography
RANSAC_REPROJ_THRESHOLD = 1.0          # MAGSAC++ 重投影誤差閾值 (像素)
MIN_INLIER_COUNT = 10                  # MAGSAC++ 至少需要找到的內點數量


def rootsift(des):
    """對 SIFT 描述符應用 RootSIFT 轉換"""
    if des is None:
        return None
    des = np.float32(des)
    # L1 正規化
    des /= (np.sum(np.abs(des), axis=1, keepdims=True) + 1e-7)
    # 開平方根
    des = np.sqrt(des)
    return des


def align_images(img1_bytes: bytes, img2_bytes: bytes):
    """
    使用 SIFT, FLANN, 和 MAGSAC++ 將 img1 對齊到 img2

    Args:
        img1_bytes: 要對齊的圖片的位元組資料。
        img2_bytes: 參考圖片的位元組資料。

    Returns:
        np.ndarray | None:
            -回傳對齊後的圖片 1 (BGR 格式的 NumPy 陣列)
    """
    try:
        # 從位元組資料解碼圖片
        img1_color = cv2.imdecode(np.frombuffer(
            img1_bytes, np.uint8), cv2.IMREAD_COLOR)
        img2_color = cv2.imdecode(np.frombuffer(
            img2_bytes, np.uint8), cv2.IMREAD_COLOR)

        # 轉換為灰階
        img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
        h1, w1 = img1_gray.shape
        h2, w2 = img2_gray.shape
        print(f"圖片解碼成功。圖1: {w1}x{h1}, 圖2: {w2}x{h2}")

        # 初始化 SIFT 偵測器
        sift = cv2.SIFT_create()

        # 偵測關鍵點並計算 RootSIFT 描述符
        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)
        print(f"圖片 1: {len(kp1)} 個關鍵點。圖片 2: {len(kp2)} 個關鍵點")

        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
            return None, f"錯誤：偵測到的關鍵點不足 (圖片1: {len(kp1)}, 圖片2: {len(kp2)})。"

        des1 = rootsift(des1)
        des2 = rootsift(des2)

        # 顯示並儲存偵測到的關鍵點
        img1_keypoints = cv2.drawKeypoints(
            img1_color, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img2_keypoints = cv2.drawKeypoints(
            img2_color, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(os.path.join(
            OUTPUT_DIR, "img1_keypoints.png"), img1_keypoints)
        cv2.imwrite(os.path.join(
            OUTPUT_DIR, "img2_keypoints.png"), img2_keypoints)

        # 特徵匹配 (FLANN)
        FLANN_INDEX_KDTREE = 1  # (KD-Tree 演算法)
        # 構建的 KD-Tree 數量，越多越快，但記憶體使用量增加
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # 增加 checks 會提高精度，但降低速度
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # 確保描述符是 float32 類型以供 FLANN 使用
        if des1 is not None:
            des1 = np.float32(des1)
        if des2 is not None:
            des2 = np.float32(des2)

        matches = flann.knnMatch(des1, des2, k=2)  # 尋找最近的 2 個鄰居
        print(f"FLANN 找到 {len(matches)} 組原始匹配")

        # Lowe's Ratio Test (篩選良好匹配)
        good_matches = []
        # 在解包前檢查 matches 列表是否包含成對的匹配項
        valid_match_pairs = [m_n for m_n in matches if len(m_n) == 2]
        if not valid_match_pairs:
            print("knnMatch 後未找到有效的匹配對")
        else:
            for m, n in valid_match_pairs:
                # 如果第一個匹配點的距離小於第二個匹配點距離的 RATIO_THRESHOLD 倍
                if m.distance < RATIO_THRESHOLD * n.distance:
                    good_matches.append(m)  # 保留這個好的匹配點
        print(f"Lowe's Ratio Test 後剩下 {len(good_matches)} 組良好匹配")
        if good_matches:
            img_good_matches = cv2.drawMatches(
                img1_color, kp1, img2_color, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(os.path.join(
                OUTPUT_DIR, "good_matches.png"), img_good_matches)

        # 估計 Homography 矩陣 (MAGSAC++)
        if len(good_matches) >= MIN_GOOD_MATCH_COUNT:  # 如果有足夠的良好匹配點
            # 從良好匹配中提取對應點的座標
            pts1 = np.float32(
                [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            pts2 = np.float32(
                [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 使用 MAGSAC++ 尋找 Homography 矩陣 (H) 和內點遮罩 (mask)
            H, mask = cv2.findHomography(
                pts1, pts2, cv2.USAC_MAGSAC, RANSAC_REPROJ_THRESHOLD)
            inlier_count = np.sum(mask)  # 計算內點數量
            print(f"MAGSAC++ 找到 {inlier_count} 個內點")

            if inlier_count >= MIN_INLIER_COUNT:  # 如果內點數量足夠
                # 顯示並儲存內點匹配
                matchesMask = mask.ravel().tolist()  # 將遮罩轉換為列表
                img_magsac_inliers = cv2.drawMatches(img1_color, kp1, img2_color, kp2, good_matches,
                                                     None, matchesMask=matchesMask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imwrite(os.path.join(
                    OUTPUT_DIR, "MAGSAC++_inlier_matches.png"), img_magsac_inliers)

                # 使用 Homography 矩陣 H 對圖片 1 進行透視變換
                aligned_img1 = cv2.warpPerspective(
                    img1_color, H, (w2, h2), flags=cv2.INTER_CUBIC)  # 使用 CUBIC 插值

                # 顯示並儲存圖像對齊
                cv2.imwrite(os.path.join(
                    OUTPUT_DIR, "aligned_image1.png"), aligned_img1)
                # 將參考圖片和對齊後的圖片混合，用於視覺比較
                blended_image = cv2.addWeighted(
                    img2_color, 0.5, aligned_img1, 0.5, 0.0)
                cv2.imwrite(os.path.join(
                    OUTPUT_DIR, "blended_result.png"), blended_image)

                return aligned_img1, img2_color
            else:
                # 如果內點不足
                print(f"內點數量不足 ({inlier_count}/{MIN_INLIER_COUNT})。對齊效果可能不佳")
                # 即使內點不足，也嘗試進行變換
                aligned_img1 = cv2.warpPerspective(
                    img1_color, H, (w2, h2), flags=cv2.INTER_CUBIC)
                # 儲存低內點數的對齊結果以供檢查
                cv2.imwrite(os.path.join(
                    OUTPUT_DIR, "aligned_image1_low_inliers.png"), aligned_img1)

                return aligned_img1, img2_color

        else:
            # 良好匹配點不足以估計 Homography
            print(
                f"找不到足夠的良好匹配點 ({len(good_matches)}/{MIN_GOOD_MATCH_COUNT}) 來估計 Homography")
            return None
    except Exception as e:
        print(f"對齊過程中發生錯誤：{e}")
        return None
def crop_images(img1_aligned: np.ndarray, img2_ref: np.ndarray):
    """
    裁剪兩張圖片，去除 img1_aligned 因 warpPerspective 產生的黑色邊框
    裁剪區域是 img1_aligned 中最大不含黑色像素的矩形區域

    Args:
        img1_aligned: 經過 warpPerspective 變換後的圖片 
        img2_ref: 原始的參考圖片。

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]:
            - (裁剪後的 aligned 圖片, 裁剪後的 ref 圖片)
    """

    # 找到非黑色像素區域
    gray_aligned = cv2.cvtColor(img1_aligned, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_aligned, 0, 255, cv2.THRESH_BINARY)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "crop_mask.png"), mask)

    # mask 中最大的輪廓
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 尋找面積最大的輪廓
    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)
    print(f"找到的有效裁剪區域: x={x}, y={y}, w={w}, h={h}")

    # 檢查邊界框是否有效 (寬度和高度必須大於 0)
    if w <= 0 or h <= 0:
        print(f"邊界框無效 (寬={w}, 高={h})，無法進行裁剪。")
        return None, None

    # 裁剪兩張圖片
    cropped_aligned = img1_aligned[y:y+h, x:x+w]
    # 確保裁剪區域不超出參考圖像的邊界
    ref_h, ref_w = img2_ref.shape[:2]
    crop_y_end = min(y + h, ref_h)
    crop_x_end = min(x + w, ref_w)
    cropped_ref = img2_ref[y:crop_y_end, x:crop_x_end]

    print(f"圖像尺寸: {cropped_aligned.shape}")
    return cropped_aligned, cropped_ref

def shrink(image):
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    #cv2.imwrite("mask.png", mask)
    # 找輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("找不到白色區域")

    # 選最大輪廓（可能就是白底）
    largest_contour = max(contours, key=cv2.contourArea)

    # 近似成四邊形
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) != 4:
        raise ValueError(f"偵測到的角點不是四個，而是 {len(approx)}")

    # 排序四個角點為 top-left, top-right, bottom-right, bottom-left
    points = approx.reshape(4, 2)
    #print(f"偵測到的四個角點：{points}")
    x=[]
    y=[]
    for i in range(4):
        x.append(points[i][0])
        y.append(points[i][1])
    x = sorted(x)
    y = sorted(y)
    print(f"排序後的 x: {x}")
    print(f"排序後的 y: {y}")
    return image[y[1]:y[2], x[1]:x[2]]  # 裁剪區域 

# 測試
if __name__ == "__main__":
    start_time = time.time()
    img1_path = "../../../assets/test_images/1.png"  # 要對齊的圖片
    img2_path = "../../../assets/test_images/2.png"  # 參考圖片

    print(f"測試圖片 1: {img1_path}")
    print(f"測試圖片 2: {img2_path}")

    # 讀取測試圖片為位元組
    try:
        with open(img1_path, 'rb') as f1, open(img2_path, 'rb') as f2:
            img1_bytes = f1.read()
            img2_bytes = f2.read()
        print("成功讀取測試圖片位元組。")
    except Exception as e:
        print(f"讀取測試圖片時發生錯誤：{e}")
        exit()

    aligned_image_np, ref_image_np = align_images(
        img1_bytes,
        img2_bytes
    )
    

    print(f"對齊後圖片的形狀：{aligned_image_np.shape}，類型：{aligned_image_np.dtype}")
    cv2.imwrite(os.path.join(
        OUTPUT_DIR, "final_aligned_test_output.png"), aligned_image_np)

    if aligned_image_np is None or ref_image_np is None:
        print("無法進行裁剪。")
    else:
        cropped_aligned, cropped_ref = crop_images(aligned_image_np, ref_image_np)

        if cropped_aligned is not None and cropped_ref is not None:

            cv2.imwrite(os.path.join(
                OUTPUT_DIR, "cropped_aligned.png"), cropped_aligned)
            cv2.imwrite(os.path.join(
                OUTPUT_DIR, "cropped_ref.png"), cropped_ref)

            blended_cropped = cv2.addWeighted(
                cropped_ref, 0.5, cropped_aligned, 0.5, 0.0)
            cv2.imwrite(os.path.join(
                OUTPUT_DIR, "blended_cropped.png"), blended_cropped)
        else:
            print("圖像裁剪失敗。")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"執行時間：{execution_time:.2f} 秒")
