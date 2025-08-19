# backend/core/alignment.py

import cv2
import numpy as np
import base64
from typing import Dict, Any, Generator, Tuple, Optional

# --- 輔助函數 ---


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


def draw_matches_preview(img1, kp1, img2, kp2, matches):
    """繪製特徵匹配線的預覽圖"""
    preview_matches = sorted(matches, key=lambda x: x.distance)[:20]
    return cv2.drawMatches(img1, kp1, img2, kp2, preview_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def encode_to_base64(img_np: np.ndarray) -> str:
    """將 numpy 圖像編碼為 base64 字串"""
    _, buffer = cv2.imencode('.png', img_np)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"


def shrink_crop(image: np.ndarray) -> Optional[np.ndarray]:
    """
    你提供的 shrink 函數，稍作修改以提高穩健性。
    優先嘗試用四點裁切，失敗則返回 None。
    """
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None  # 找不到輪廓

    largest_contour = max(contours, key=cv2.contourArea)

    # 嘗試近似成四邊形
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        points = approx.reshape(4, 2)
        # 根據 x 坐標排序找到左右，再根據 y 坐標找到上下
        rect = np.zeros((4, 2), dtype="float32")
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]  # Top-left
        rect[2] = points[np.argmax(s)]  # Bottom-right

        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]  # Top-right
        rect[3] = points[np.argmax(diff)]  # Bottom-left

        (tl, tr, br, bl) = rect
        # 計算寬度和高度
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        if maxWidth > 0 and maxHeight > 0:
            # 進行透視變換來"拉直"矩形
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            return warped

    # 如果 shrink 失敗，返回 None，讓主流程降級處理
    return None


def bounding_box_crop(img1_aligned: np.ndarray, img2_ref: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    你提供的 crop_images 函數，作為降級方案。
    """
    gray_aligned = cv2.cvtColor(img1_aligned, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_aligned, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    if w <= 0 or h <= 0:
        return None, None

    cropped_aligned = img1_aligned[y:y+h, x:x+w]
    ref_h, ref_w, _ = img2_ref.shape
    crop_y_end = min(y + h, ref_h)
    crop_x_end = min(x + w, ref_w)
    cropped_ref = img2_ref[y:crop_y_end, x:crop_x_end]

    return cropped_aligned, cropped_ref

# --- 主要的生成器函數 ---


def align_images_pipeline(
    img1_bytes: bytes,
    img2_bytes: bytes,
    min_match_count: int = 10,
    lowe_ratio: float = 0.6,  # 0.7
    magsac_threshold: float = 1.0  # 4.0
) -> Generator[Dict[str, Any], None, None]:  # 注意：生成器本身不返回值

    img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)
    original_img1 = img1.copy()
    original_img2 = img2.copy()
    # 轉換為灰階
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    try:
        # --- 細項 1: SIFT + RootSIFT ---
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)

        des1 = rootsift(des1)
        des2 = rootsift(des2)

        if des1 is None or des2 is None or len(kp1) < min_match_count or len(kp2) < min_match_count:
            raise ValueError(f"提取到的特徵點不足 (圖一: {len(kp1)}, 圖二: {len(kp2)})。")

        img1_kp = cv2.drawKeypoints(
            original_img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img2_kp = cv2.drawKeypoints(
            original_img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        yield {
            "name": "RootSIFT 特徵提取",
            "status": "completed",
            "previews": [encode_to_base64(img1_kp), encode_to_base64(img2_kp)],
            "text": f"圖一找到 {len(kp1)} 個特徵點，圖二找到 {len(kp2)} 個特徵點。"
        }

        # --- 細項 2: FLANN 特徵匹配 ---
        FLANN_INDEX_KDTREE = 1  # (KD-Tree 演算法)
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # 增加 checks 會提高精度，但降低速度
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)   # 尋找最近的 2 個鄰居

        # Lowe's Ratio Test (篩選良好匹配)
        good_matches = [
            m for m, n in matches if m is not None and n is not None and m.distance < lowe_ratio * n.distance]
        img_matches = draw_matches_preview(
            original_img1, kp1, original_img2, kp2, good_matches)

        yield {
            "name": "FLANN 特徵匹配",
            "status": "completed",
            "preview_match": encode_to_base64(img_matches),
            "text": f"找到 {len(good_matches)} 個好的匹配點。"
        }

        # --- 細項 3: MAGSAC++ 變換與裁切 ---
        if len(good_matches) <= min_match_count:
            raise ValueError(
                f"好的匹配點不足 ({len(good_matches)}/{min_match_count})，無法進行可靠的對齊。")

        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        # 使用 MAGSAC++ 尋找 Homography 矩陣 (H) 和內點遮罩 (mask)
        H, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.USAC_MAGSAC, magsac_threshold)
        if H is None:
            raise ValueError("MAGSAC++ 無法計算出穩定的變換矩陣。")

        h, w, _ = original_img2.shape
        img1_aligned = cv2.warpPerspective(
            original_img1, H, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        crop_text = ""
        # 優先嘗試 shrink_crop
        img1_shrunk = shrink_crop(img1_aligned)
        if img1_shrunk is not None:
            crop_text = "使用四點變換進行精確裁切。"
            # 如果 shrink 成功，我們需要重新計算 img2 的對應區域
            # 為了簡化，我們先用 Bounding Box 來裁切 img2
            _, _, w_shrunk, h_shrunk = cv2.boundingRect(max(cv2.findContours(cv2.cvtColor(
                img1_aligned, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea))
            img1_final = cv2.resize(
                img1_shrunk, (w_shrunk, h_shrunk))  # 將拉直的圖縮放回原比例
            _, img2_final = bounding_box_crop(img1_aligned, original_img2)

        else:
            # shrink 失敗，降級為 bounding_box_crop
            crop_text = "四點裁切失敗，降級為邊界框裁切。"
            img1_final, img2_final = bounding_box_crop(
                img1_aligned, original_img2)

        if img1_final is None or img2_final is None:
            raise ValueError("所有裁切方法均失敗。")

        yield {
            "name": "對齊與裁切",
            "status": "completed",
            "previews": [encode_to_base64(img1_final), encode_to_base64(img2_final)],
            "text": crop_text + f" 最終圖像尺寸: {img1_final.shape[1]}x{img1_final.shape[0]}。"
        }

        yield {"final_result": (img1_final, img2_final, True)}

    except Exception as e:
        yield {
            "name": "對齊失敗",
            "status": "failed",
            "text": str(e)
        }
        h1, w1, _ = original_img1.shape
        h2, w2, _ = original_img2.shape
        target_h, target_w = min(h1, h2), min(w1, w2)
        img1_resized = cv2.resize(original_img1, (target_w, target_h))
        img2_resized = cv2.resize(original_img2, (target_w, target_h))
        # **失敗時，也 yield 一個帶有最終結果的特殊字典**
        yield {
            "final_result": (img1_resized, img2_resized, False)
        }
