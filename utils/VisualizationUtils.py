import cv2
import numpy as np


def save_img_cv2(path, img, expand=False, img_min=None, img_max=None):
    if expand:
        img_min, img_max = np.min(img[img != 0]), np.max(img[img != 0])

    if img_min is not None:
        img[img != 0] = (img[img != 0] - img_min) / (img_max - img_min)

    img = img.clip(0, 1)
    img = (img * 255).astype(np.uint8)
    if expand or img_min is not None:
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)
    return img_min, img_max


def save_heatmap_over_img(path, true, heatmap):
    img = true * 255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # heatmap = np.transpose(heatmap, (1,2,0))
    heatmap = heatmap  # / np.max(heatmap)
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img = cv2.addWeighted(heatmap, 0.5, img, 0.5, 0)
    cv2.imwrite(path, img)
