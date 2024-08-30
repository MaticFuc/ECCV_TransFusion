import glob

import cv2
import numpy as np
import tifffile
import torch
import torchvision
from PIL import Image

from utils.geo_utils import *


def get_faulty_imgs(dataset, files):
    a = []
    for f in files:
        m = get_mask_filename(dataset, f)
        if isinstance(m, list):
            mask_full = None
            for mi in m:
                mask = cv2.imread(mi, cv2.IMREAD_GRAYSCALE) / 255
                if mask_full is None:
                    mask_full = mask
                else:
                    mask_full = np.logical_or(mask_full, mask)
            mask = mask_full * 255
            mask = mask.astype(np.uint8)
        else:
            mask = cv2.imread(m, cv2.IMREAD_GRAYSCALE)

        if mask is not None and np.max(mask) > 0:
            a.append(False)
        else:
            a.append(True)
    a = np.array(a)
    return a


def get_files(dataset, test, path):
    if dataset == "visa":
        if test:
            p = list([p for p in glob.glob(f"{path}/*/*.JPG")])
            is_faulty = get_faulty_imgs(dataset, np.array(p))
            return list(p), is_faulty
        return list([p for p in glob.glob(f"{path}/*.JPG")])
    elif dataset == "mvtec":
        if test:
            p = list([p for p in glob.glob(f"{path}/*/*.png")])
            is_faulty = get_faulty_imgs(dataset, np.array(p))
            return list(p), is_faulty

        return list([p for p in glob.glob(f"{path}/*.png")])
    elif dataset == "mvtec3d":
        if test:
            p = list([p for p in glob.glob(f"{path}/*/rgb/*.png")])
            is_faulty = get_faulty_imgs(dataset, np.array(p))
            return list(p), is_faulty
        return list([p for p in glob.glob(f"{path}/rgb/*.png")])


def get_depth_files(test, path):
    if test:
        p = list([p for p in glob.glob(f"{path}/*/xyz/*.tiff")])
        return list(p)

    return list([p for p in glob.glob(f"{path}/xyz/*.tiff")])


def get_mask_filename(dataset, filename):
    if dataset == "visa":
        return filename.replace("test", "ground_truth").replace(".JPG", ".png")
    elif dataset == "mvtec":
        return filename.replace("test", "ground_truth").replace(".png", "_mask.png")
    elif dataset == "mvtec3d":
        return filename.replace("rgb", "gt")


def get_max_min_depth_img(image_path):
    image = tifffile.imread(image_path).astype(np.float32)
    image_t = (
        np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
    )
    image = image_t[:, :, 2]
    zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
    im_max = np.max(image)
    im_min = np.min(image * (1.0 - zero_mask) + 1000 * zero_mask)
    return im_min, im_max


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        datasetParameters: dict,
        test: bool,
    ):
        super().__init__()
        self.image_size = 256
        self.test = test

        self.category = datasetParameters["category"]
        self.dataset_type = datasetParameters["dataset_type"]

        postfix = "test/" if test else "train/good/"
        path = f"{path}/{self.category}/{postfix}"

        out = get_files(self.dataset_type, self.test, path)
        if isinstance(out, tuple):
            self._files = out[0]
            self.is_faulty = np.logical_not(out[1])
        else:
            self._files = out
            self.is_faulty = [False] * len(self._files)
        idx = np.argsort(self._files)
        self.is_faulty = list(np.array(self.is_faulty)[idx])
        self._files.sort()

        self.global_max, self.global_min = 1, 0
        self.d = "d" in datasetParameters["mode"]
        self.rgb = "rgb" in datasetParameters["mode"]
        self.depth_files = None
        if self.d:
            self.depth_files = get_depth_files(self.test, path)
            self.depth_files.sort()
            if not test:
                for image_path in self.depth_files:
                    im_min, im_max = get_max_min_depth_img(image_path)
                    self.global_min = min(self.global_min, im_min)
                    self.global_max = max(self.global_max, im_max)
                self.global_min = self.global_min * 0.9
                self.global_max = self.global_max * 1.1
            else:
                self.global_max = datasetParameters["global_max"]
                self.global_min = datasetParameters["global_min"]

    def __len__(self):
        return len(self._files)

    def get_image(self, file, mask=False, size=None):
        img = Image.open(file)
        size = self.image_size if size is None else size
        img = torchvision.transforms.Resize((size, size))(img)
        if mask:
            return torch.FloatTensor(np.array(img))
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.stack((img, img, img), axis=2)

        img = (img / 255.0 - 0.5) * 2
        img = img.transpose((2, 0, 1))

        return torch.FloatTensor(img)

    def get_depth_image(self, file, size=None):
        depth_img = tifffile.imread(file).astype(np.float32)
        size = self.image_size if size is None else size
        depth_img = cv2.resize(
            depth_img, (size, size), 0, 0, interpolation=cv2.INTER_NEAREST
        )
        depth_img = np.array(depth_img)

        image = depth_img
        image_t = (
            np.array(image)
            .reshape((image.shape[0], image.shape[1], 3))
            .astype(np.float32)
        )
        image = image_t[:, :, 2]

        zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
        plane_mask = get_plane_mask(image_t)  # 0 is background, 1 is foreground
        plane_mask[:, :, 0] = plane_mask[:, :, 0] * (1.0 - zero_mask)
        plane_mask = fill_plane_mask(plane_mask)

        image = fill_depth_map(image)
        zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))

        image = (image - self.global_max) / (self.global_max - self.global_min)

        image = image * (1.0 - zero_mask)

        image = np.expand_dims(image, 2)

        depth_img = image

        return torch.FloatTensor(depth_img.transpose((2, 0, 1))), torch.FloatTensor(
            np.squeeze(plane_mask)
        )

    def __getitem__(self, index: int):
        img = self.get_image(self._files[index], mask=False)

        has_anom = self.is_faulty[index]
        if has_anom:
            mask = get_mask_filename(self.dataset_type, self._files[index])
            mask = self.get_image(mask, mask=True)
        else:
            size = img.shape[1]
            mask = torch.FloatTensor(np.zeros((size, size)))

        sample = {"image": img, "mask": mask, "index": index, "is_anomaly": has_anom}

        if self.depth_files is not None:
            d_img, plane_mask = self.get_depth_image(self.depth_files[index])
            sample["plane_mask"] = plane_mask

        if self.d and self.rgb:
            full_img = torch.cat([img, d_img], dim=0)
            sample["image"] = full_img
        elif self.d:
            sample["image"] = d_img
        return sample
