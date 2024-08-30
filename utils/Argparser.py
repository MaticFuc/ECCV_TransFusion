import argparse


def get_argparser():
    parser = argparse.ArgumentParser(description="TransFusion")

    parser.add_argument(
        "-c",
        "--choice",
        type=str,
        default="train",
    )
    parser.add_argument(
        "-r", "--run-name", type=str, default="TransFusion_Test"
    )  # Change for a more descriptive name
    parser.add_argument(
        "-d",
        "--data-path",
        type=str,
        default="/path/to/your/dataset/",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="./experiments/",
    )
    parser.add_argument(
        "-ds",
        "--dataset",
        type=str,
        default="mvtec3d",
        choices=["mvtec3d", "mvtec", "visa"],
    )
    parser.add_argument(
        "--mode", type=str, default="rgbd", choices=["rgb", "d", "rgbd"]
    )
    parser.add_argument("--fg-mask-path", type=str, default="./fg_masks/")
    parser.add_argument("--no-fg-masks", default=True, action="store_false")
    parser.add_argument("--dtd-path", type=str, default="/storage/datasets/DTD/images/")
    parser.add_argument(
        "--category", type=str, default="all"
    )  # "all" option will run for all categories
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--visualize", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=3)

    parser.add_argument("--epoch-num", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--step-size", type=int, default=800)
    parser.add_argument("--gamma", type=float, default=0.1)

    parser.add_argument("--diffusion-num-steps", type=int, default=20)

    parser.add_argument(
        "--unet-channel-num", type=int, default=64
    )  # Double in the RGBD mode
    # Change the params of these three lists here, IDK how to pass them via CMD
    parser.add_argument("--unet-ch-mults", type=list, default=[1, 2, 2, 2])
    parser.add_argument("--unet-attn", type=list, default=[False, False, True, True])
    parser.add_argument("--unet-group-conv", type=int, default=[True, True, True, True])
    parser.add_argument("--unet-n-blocks", type=int, default=2)

    parser.add_argument("--img-size", type=int, default=256)

    parser.add_argument("--eval-kernel-size", type=int, default=7)
    parser.add_argument("--eval-w", type=float, default=0.95)
    parser.add_argument(
        "--eval-epoch-num", type=int, default=750
    )  # Evaluation will happen every N epochs, only the model from the last epoch will be saved

    return parser
