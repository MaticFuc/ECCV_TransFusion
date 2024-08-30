import os
import warnings
from os import path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from scipy import signal
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from utils import Argparser, EvaluationUtils, ParsingUtils, VisualizationUtils


def train(args):

    (
        PATH,
        trainset,
        testset,
        net,
        optimizer,
        scheduler,
        mode,
    ) = ParsingUtils.parse_args(args)

    start_epoch = 0
    if args.resume:
        with open(PATH + f"models/{args.category}_epoch.txt", "r") as f:
            l = f.read()
            start_epoch = int(l) + 1
        state_dict = torch.load(PATH + f"models/{args.category}_state_dict.pkl")
        optimizer.load_state_dict(state_dict["opt_state_dict"])
        scheduler.load_state_dict(state_dict["sched_state_dict"])
        net.load_state_dict(state_dict["net_state_dict"])
    os.makedirs(PATH + "models", exist_ok=True)

    batch_size = args.bs

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    epochs = args.epoch_num
    evaluate_step = args.eval_epoch_num

    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        net.train()
        with tqdm(
            total=len(trainset),
            desc="Epoch: " + str(epoch) + "/" + str(epochs),
            unit="img",
        ) as prog_bar:
            for i, data in enumerate(trainloader, 0):
                inputs = data["image"].cuda()
                plane_mask = None
                if "plane_mask" in data:
                    plane_mask = data["plane_mask"].cuda()
                optimizer.zero_grad()

                loss = net.loss(inputs, idx=data["index"], plane_mask=plane_mask)

                running_loss += loss.item()

                loss.backward()
                optimizer.step()
                running_loss = running_loss + loss.data.cpu().detach().numpy()

                prog_bar.set_postfix(
                    **{
                        "loss": running_loss / ((i + 1) * batch_size),
                        "lr": scheduler.get_last_lr()[0],
                    }
                )
                prog_bar.update(batch_size)

        if epoch % evaluate_step == 0:
            net.eval()
            with open(PATH + f"models/{args.category}_epoch.txt", "w") as f:
                torch.save(
                    {
                        "net_state_dict": net.state_dict(),
                        "opt_state_dict": optimizer.state_dict(),
                        "sched_state_dict": scheduler.state_dict(),
                    },
                    PATH + f"models/{args.category}_state_dict.pkl",
                )
                f.write(f"{epoch}")
            experiment = {
                "num_steps": net.steps - 1,
                "kernel_size": args.eval_kernel_size,
                "weight": args.eval_w,
                "epoch": epoch,
                "mode": mode,
                "img_size": args.img_size,
                "category": args.category,
                "save_imgs": args.visualize,
            }
            evaluate(
                PATH,
                testset,
                net,
                experiment,
            )
        scheduler.step()
    torch.save(net.state_dict(), PATH + f"models/{args.category}_model.pkl")


def test(args):

    (PATH, _, testset, net, _, _, mode) = ParsingUtils.parse_args(args)

    experiment = {
        "num_steps": net.steps - 1,
        "kernel_size": args.eval_kernel_size,
        "weight": args.eval_w,
        "epoch": args.epoch_num,
        "mode": mode,
        "img_size": args.img_size,
        "category": args.category,
        "save_imgs": args.visualize,
    }
    auroc, aupro = evaluate(PATH, testset, net, experiment, load_weights=True)
    return auroc, aupro


def evaluate(PATH, testset, net, parameters, load_weights=False):
    category = parameters["category"]
    save_imgs = parameters["save_imgs"]
    if save_imgs:
        os.makedirs(f"{PATH}visualizations/{category}/", exist_ok=True)
    if load_weights:
        net.load_state_dict(torch.load(PATH + f"models/{category}_model.pkl"))
    net.eval()

    batch_size = 1
    epoch = parameters["epoch"]

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    predScore = []
    trueScore = []

    k = parameters["kernel_size"]
    kern = np.ones((k, k)) / (k**2)
    w = parameters["weight"]
    img_size = parameters["img_size"]

    all_masks = np.zeros((len(testset), img_size, img_size))
    true_masks = np.zeros((len(testset), img_size, img_size))

    with torch.no_grad():
        with tqdm(
            total=len(testset),
            desc=f"Evaluation - {category} Epoch {epoch}: ",
            unit="img",
        ) as prog_bar:
            for j, data in enumerate(testloader, 0):

                (
                    reconstructed,
                    mask_disc,
                    mask_recon,
                ) = EvaluationUtils.calculate_transfussion_results(
                    net,
                    testset,
                    data["index"],
                )

                mask_disc = signal.convolve2d(
                    mask_disc.squeeze(), kern, mode="same"
                ).squeeze()
                mask_recon = signal.convolve2d(
                    mask_recon.squeeze(), kern, mode="same"
                ).squeeze()
                mask_final = w * mask_disc + (1 - w) * mask_recon

                mask = data["mask"].numpy().squeeze() > 0.5
                mask = mask.astype("int")

                all_masks[j, :, :] = mask_final
                true_masks[j, :, :] = mask
                if save_imgs:
                    img_idx = data["index"].cpu().numpy()[0]
                    img = data["image"].cpu().numpy().squeeze() / 2 + 0.5
                    depth = "d" in parameters["mode"]
                    rgb = "rgb" in parameters["mode"]

                    if rgb and depth:
                        depth_recon = reconstructed[3, :, :]
                        reconstructed = reconstructed[:3, :, :]
                        depth_true = (img[3, :, :] - 0.5) * 2
                        img = img[:3, :, :]
                    elif depth:
                        depth_recon = reconstructed[0, :, :]
                        depth_true = (img[0, :, :] - 0.5) * 2

                    VisualizationUtils.save_img_cv2(
                        f"{PATH}visualizations/{category}/{img_idx}_true_mask.png", mask
                    )
                    VisualizationUtils.save_img_cv2(
                        f"{PATH}visualizations/{category}/{img_idx}_mask.png",
                        mask_final,
                    )
                    if rgb:
                        VisualizationUtils.save_img_cv2(
                            f"{PATH}visualizations/{category}/{img_idx}_reconstructed.png",
                            reconstructed.transpose((1, 2, 0)),
                        )
                        VisualizationUtils.save_img_cv2(
                            f"{PATH}visualizations/{category}/{img_idx}_true.png",
                            img.transpose((1, 2, 0)),
                        )
                        VisualizationUtils.save_heatmap_over_img(
                            f"{PATH}visualizations/{category}/{img_idx}_heatmap.png",
                            img.transpose((1, 2, 0)),
                            mask_final,
                        )
                    if depth:

                        img_min, img_max = VisualizationUtils.save_img_cv2(
                            f"{PATH}visualizations/{category}/{img_idx}_true_depth.png",
                            depth_true,
                            expand=True,
                        )
                        VisualizationUtils.save_img_cv2(
                            f"{PATH}visualizations/{category}/{img_idx}_reconstructed_depth.png",
                            depth_recon,
                            img_min=img_min,
                            img_max=img_max,
                        )

                predScore.append(np.max(mask_final))
                trueScore.append(np.max(mask))
                prog_bar.update(batch_size)

    rocScoreImg = roc_auc_score(trueScore, predScore)
    pro = EvaluationUtils.compute_pro(all_masks, true_masks)
    run_name = PATH.split("/")[-2]

    print(
        f"Final score - epoch {epoch} - {run_name}: AUROC: {rocScoreImg}, AUPRO: {pro}"
    )

    df = {
        "Category": [testset.category],
        "Epoch": [epoch],
        "Weight": [round(parameters["weight"], 2)],
        "Kernel Size": [k],
        "Roc Img": [rocScoreImg],
        "PRO Img": [pro],
    }
    df = pd.DataFrame(data=df)
    csv_mode = "a"
    if not path.isfile(f"{PATH}output.csv"):
        csv_mode = "w"
    df.to_csv(
        f"{PATH}output.csv",
        mode=csv_mode,
        index=False,
        header=csv_mode == "w",
    )
    return rocScoreImg, pro


if __name__ == "__main__":
    parser = Argparser.get_argparser()
    args = parser.parse_args()

    categories = {
        "mvtec3d": [
            "cable_gland",
            "bagel",
            "cookie",
            "carrot",
            "dowel",
            "foam",
            "peach",
            "potato",
            "tire",
            "rope",
        ],
        "mvtec": [
            "capsule",
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
            "wood",
        ],
        "visa": [
            "candle",
            "capsules",
            "cashew",
            "chewinggum",
            "fryum",
            "macaroni1",
            "macaroni2",
            "pcb1",
            "pcb2",
            "pcb3",
            "pcb4",
            "pipe_fryum",
        ],
    }

    if args.category == "all":
        categories = categories[args.dataset]
        categories.sort()
    else:
        categories = [args.category]

    auroc_all, aupro_all = 0, 0
    if args.choice == "train":
        for category in categories:
            setattr(args, "category", category)
            train(args)
    elif args.choice == "test":
        for category in categories:
            setattr(args, "category", category)
            auroc, aupro = test(args)
            auroc_all += auroc
            aupro_all += aupro
        auroc_all /= len(categories)
        aupro_all /= len(categories)
        print(
            f"AVG AUROC: {round(auroc_all*100, 2)}, AVG AUPRO: {round(aupro_all*100, 2)}"
        )
