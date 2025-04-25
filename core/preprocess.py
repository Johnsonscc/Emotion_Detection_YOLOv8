import cv2
import numpy as np

def preprocess_image(image):
    if isinstance(image, str):  # 处理文件路径
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 4:  # 如果有batch维度
            image = image[0]
        if image.shape[2] == 1:    # 如果是灰度图
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # 如果是RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image
