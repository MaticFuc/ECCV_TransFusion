from glob import glob

import cv2
import numpy as np
import torch


class BGMaskDataset(torch.utils.data.Dataset):
    def __init__(self, path, image_size):
        super().__init__()
        path = path
        self.image_size = image_size

        self._files = glob(f"{path}*.png")
        if len(self._files) == 0:
            self._files = glob(f"{path}*.JPG")
        if len(self._files) == 0:
            path = "/".join(path.split("/")[:-3])
            print(path)
            self._files = glob(f"{path}/good/rgb/*.png")
        self._files.sort()

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index: int):
        image_path = self._files[index]
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255
        image = (image > 0.5).astype(float)
        kernel = np.ones((5, 5), np.uint8)
        image = cv2.dilate(image, kernel)
        image = torch.FloatTensor(image)
        return np.array(image)
