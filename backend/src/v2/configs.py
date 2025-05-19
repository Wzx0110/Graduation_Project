class MainConfig:
    """主流程的配置參數"""
    IMAGE_A_PATH = "../../../assets/test_images/1.png"
    COMMON_TARGET_SIZE = (512, 512)
    # semantic_segmentation.py
    SEMANTIC_MODEL_NAME = "shi-labs/oneformer_cityscapes_swin_large"
    CITYSCAPES_PALETTE = [
        [128, 64, 128],    # 0: road (道路) - 紫紅色
        [244, 35, 232],    # 1: sidewalk (人行道) - 洋紅色
        [70, 70, 70],      # 2: building (建築物) - 深灰色
        [102, 102, 156],   # 3: wall (牆) - 岩藍色
        [190, 153, 153],   # 4: fence (柵欄) - 淡棕紅色
        [153, 153, 153],   # 5: pole (電線桿/柱子) - 灰色
        [250, 170, 30],    # 6: traffic light (交通號誌燈) - 橘黃色
        [220, 220, 0],     # 7: traffic sign (交通標誌) - 黃色
        [107, 142, 35],    # 8: vegetation (植被) - 橄欖綠
        [152, 251, 152],   # 9: terrain (地形/土地) - 淡綠色
        [70, 130, 180],    # 10: sky (天空) - 天藍色
        [220, 20, 60],     # 11: person (人) - 深紅色
        [255, 0, 0],       # 12: rider (騎士) - 紅色
        [0, 0, 142],       # 13: car (汽車) - 深藍色
        [0, 0, 70],        # 14: truck (卡車) - 海軍藍
        [0, 60, 100],      # 15: bus (巴士) - 深青藍色
        [0, 80, 100],      # 16: train (火車) - 深水藍色
        [0, 0, 230],       # 17: motorcycle (摩托車) - 藍色
        [119, 11, 32],     # 18: bicycle (自行車) - 暗紅色
        [0, 0, 0]          # void (空) - 黑色
    ]
