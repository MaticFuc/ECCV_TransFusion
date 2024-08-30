import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset.Dataset import Dataset
from model.TransFusion import TransFusion
from model.UNet import UNet


def get_optimizer(args, net):
    return optim.AdamW(net.parameters(), lr=args.lr)


def get_scheduler(args, optimizer):
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma,
    )


def parse_args(args) -> nn.Module:
    os.environ["OPENCV_LOG_LEVEL"] = "OFF"
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

    torch_seed(args.seed)
    log_dir, trainDataset = create_train_regime(args)
    testDataset = get_dataset(
        args, False, trainDataset.global_max, trainDataset.global_min
    )
    net = create_model(args)
    optimizer, scheduler = get_experiment_variables(args, net)

    return (log_dir, trainDataset, testDataset, net, optimizer, scheduler, args.mode)


def get_unet(args):
    UNetParameters = {
        "ch_mults": args.unet_ch_mults,
        "is_attn": args.unet_attn,
        "use_groups": args.unet_group_conv,
        "n_blocks": args.unet_n_blocks,
    }
    d = "d" in args.mode
    rgb = "rgb" in args.mode
    UNetParameters["image_channels"] = 3 * int(rgb) + int(d)
    UNetParameters["is_rgbd"] = UNetParameters["image_channels"] == 4
    UNetParameters["n_channels"] = (
        2 * args.unet_channel_num
        if UNetParameters["image_channels"] == 4
        else args.unet_channel_num
    )
    unet = UNet(UNetParameters)
    return unet


def get_noise_generator_parameters(args):
    params = {
        "path": args.dtd_path,
        "dataset": args.dataset,
        "category": args.category,
        "mode": args.mode,
        "use_pretrained_masks": args.no_fg_masks,
        "fg_mask_path": args.fg_mask_path,
        "img_size": args.img_size,
    }
    params["is_texture"] = args.category in [
        "carpet",
        "grid",
        "leather",
        "tile",
        "wood",
        "zipper",
    ]
    return params


def create_model(args) -> nn.Module:
    unet = get_unet(args)

    diffModelParameters = {"num_steps": args.diffusion_num_steps, "mode": args.mode}

    diffModelParameters["noise_generator_parameters"] = get_noise_generator_parameters(
        args
    )

    model = TransFusion(unet, diffModelParameters).cuda()
    return model


def get_experiment_variables(args, net):
    optimizer = get_optimizer(args, net)
    scheduler = get_scheduler(args, optimizer)

    return optimizer, scheduler


def create_train_regime(args):
    log_dir = args.log_path + "/" + args.run_name + "/"
    os.makedirs(log_dir, exist_ok=True)
    trainSet = get_dataset(args, True)
    return log_dir, trainSet


def get_dataset(
    args: dict, train: bool, global_max: float = 1.0, global_min: float = 0.0
):
    datasetParameters = {
        "size": args.img_size,
        "mode": args.mode,
        "dataset_type": args.dataset,
        "category": args.category,
        "global_max": global_max,
        "global_min": global_min,
    }

    datasetInstance = Dataset(args.data_path, datasetParameters, not train)
    return datasetInstance


def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)
