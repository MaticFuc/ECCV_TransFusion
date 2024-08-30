import glob

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch


class SyntAnomDataset(torch.utils.data.Dataset):
    def __init__(self, path, image_size) -> None:
        super().__init__()
        self.image_size = image_size
        self.dtd_samples = glob.glob(path + "/*/*.jpg")
        self.dtd_samples.sort()
        self.augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45)),
        ]

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential(
            [
                self.augmenters[aug_ind[0]],
                self.augmenters[aug_ind[1]],
                self.augmenters[aug_ind[2]],
            ]
        )
        return aug

    def __len__(self):
        return len(self.dtd_samples)

    def __getitem__(self, index):
        aug = self.randAugmenter()
        img = self.dtd_samples[index]
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(self.image_size, self.image_size))
        img = aug(image=img)
        img = img.astype(np.float32)
        img = img / 255.0
        img = img.transpose((2, 0, 1))
        img = (img - 0.5) * 2
        return {"img": img, "index": index}
