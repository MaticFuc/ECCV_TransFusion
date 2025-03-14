import torch
from segment_anything import (
    SamPredictor,
    sam_model_registry,
)
import glob
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import os
import argparse

def get_backgrounds(imgs, out_path):
    end_path = imgs[0].split("/")[-3:-1]
    end_path = "/".join(end_path)
    out_path = out_path + end_path
    os.makedirs(out_path, exist_ok=True)
    sam = setup_sam()

    for idx, img in enumerate(imgs):
        print(idx)
        img_file = cv2.imread(img)
        img_file = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
        img_file = cv2.resize(img_file, (256,256))
        
        with torch.no_grad():
            sam.set_image(img_file)
            input_points = [[0,0],[255,0],[0,255],[255,255]]
            final_mask = np.zeros((1, 256,256))
            for i, point in enumerate(input_points):
                mask, _, _ = sam.predict(point_coords=np.array([point]), point_labels=np.array([1]), multimask_output=False)
                final_mask = np.maximum(final_mask, mask)
            final_mask = 1 - final_mask
            final_mask = final_mask * 255
            final_mask = final_mask.astype(np.uint8)
            idx = img.split("/")[-1].replace(".bmp",".png")
            print(idx)
            cv2.imwrite(f"{out_path}/{idx}", final_mask.transpose((1,2,0)))

def setup_sam():
    model_path = "./pretrained_models/"
    sam_checkpoint = f"{model_path}sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamPredictor(
        sam
    )
    return mask_generator

def get_masks(args):
    path = args.path
    out_path = args.out_path
    categories = args.category
    for category in categories:
        final_path = f"{path}{category}/train/"
        print(f"{final_path}*/*{args.file_ending}")
        imgs = glob.glob(f"{final_path}*/*{args.file_ending}")
        print(imgs)
        final_out_path = out_path + f"{category}/"
        os.makedirs(final_out_path, exist_ok=True)
        get_backgrounds(imgs, final_out_path)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TransFusion")

    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="./dataset/mvtec/",
    )
    parser.add_argument(
        "-o",
        "--out-path",
        type=str,
        default="./dataset/mvtec_masks/", # The code expected the name of the folder to be <dataset_name>_masks
    )
    parser.add_argument(
        "-f",
        "--file-ending",
        type=str,
        default=".png", 
    )
    parser.add_argument(
        "-c",
        "--category",
        type=str,
        nargs='+',
        default=["capsule",
        "bottle",
        "carpet",
        "leather",
        "pill",
        "transistor",
        "tile",
        "cable",
        "zipper",
        "toothbrush",
        "metal_nut",
        "hazelnut",
        "screw",
        "grid",
        "wood",]
    )
    args = parser.parse_args()
    get_masks(args)
       