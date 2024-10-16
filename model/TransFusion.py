import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops.focal_loss import sigmoid_focal_loss

from model.SyntAnomGenerator import SyntAnomalyGenerator
from utils import MathUtils


def gather(c, t):
    c = c.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


class TransFusion(nn.Module):
    def __init__(self, model, parameters):
        super().__init__()
        self.noise_generator = SyntAnomalyGenerator(
            parameters["noise_generator_parameters"]
        )

        steps = parameters["num_steps"]
        self.mode = parameters["mode"]
        self.mask_chans = 1
        self.model = model
        self.beta = torch.linspace(0, 1, steps).cuda()
        self.steps = steps

        self.ssim_loss = MathUtils.SSIM()

    def q_sample(
        self, x0, mask, t, eps=None, add_noise=True, return_mask=False, mask_c=None
    ):

        if eps is None:
            eps, mask, mask_c = self.set_noise(x0, t)

        if not add_noise:
            b, _, x, y = eps.shape
            mask = torch.zeros((b, self.mask_chans, x, y)).cuda()
            mask_c = torch.zeros((b, self.mask_chans, x, y)).cuda()

        beta = gather(self.beta, t)
        xt = x0.clone()
        if x0.shape[1] == 4:
            x0_d = x0[:, 3, :, :].clone()
            x0_rgb = x0[:, :3, :, :].clone()

            eps_d = eps[:, 3, :, :].clone()
            eps_rgb = eps[:, :3, :, :].clone()

            mask_img = mask[:, 0, :, :].unsqueeze(1)
            mask_img = torch.tile(mask_img, (1, 3, 1, 1))
            xt[:, :3, :, :] = (
                (1 - mask_img) * x0_rgb
                + (1 - beta) * mask_img * x0_rgb
                + beta * mask_img * eps_rgb
            )
            xt[:, 3, :, :] = x0_d + beta[:, 0, :, :] * mask[:, 0, :, :] * eps_d
        elif x0.shape[1] == 3:
            mask_img = torch.tile(mask, (1, x0.shape[1], 1, 1))
            xt = (
                (1 - mask_img) * x0 + (1 - beta) * mask_img * x0 + beta * mask_img * eps
            )
        else:
            mask_img = torch.tile(mask, (1, x0.shape[1], 1, 1))
            xt = x0 + beta * mask_img * eps
        if return_mask:
            return xt, mask
        return xt, eps, mask, mask_c

    def get_next_step(self, xt, mean, mask, eps, t):
        beta_t = gather(self.beta, t)
        beta_t_min_1 = gather(self.beta, t - 1)

        diff = beta_t - beta_t_min_1
        predicted_mask = mask

        if xt.shape[1] == 4:
            predicted_mask_img = predicted_mask[:, :1, :, :].clone()
            predicted_mask_img = (
                (predicted_mask_img > 0.5).type(torch.FloatTensor).cuda()
            )
            predicted_mask_img = torch.tile(predicted_mask_img, (1, 3, 1, 1))

            predicted_mask_depth = predicted_mask[:, :1, :, :].clone()
            predicted_mask_depth = (
                (predicted_mask_depth > 0.5).type(torch.FloatTensor).cuda()
            )

        else:
            predicted_mask = (predicted_mask > 0.5).type(torch.FloatTensor).cuda()
            predicted_mask_img = torch.tile(predicted_mask, (1, xt.shape[1], 1, 1))

        new_mean = torch.zeros_like(xt)
        if xt.shape[1] == 4:
            eps_rgb = eps[:, :3, :, :].clone()
            mean_rgb = mean[:, :3, :, :].clone()
            xt_rgb = xt[:, :3, :, :].clone()

            eps_d = eps[:, 3:, :, :].clone()
            xt_d = xt[:, 3:, :, :].clone()

            new_mean[:, :3, :, :] = (
                xt_rgb
                - (diff / beta_t) * predicted_mask_img * eps_rgb
                + diff * predicted_mask_img * mean_rgb
            )
            new_mean[:, 3:, :, :] = (
                xt_d - (diff / beta_t) * predicted_mask_depth * eps_d
            )
        elif xt.shape[1] == 3:
            new_mean = (
                xt
                - (diff / beta_t) * predicted_mask_img * eps
                + diff * predicted_mask_img * mean
            )
        else:
            new_mean = xt - (diff / beta_t) * predicted_mask_img * eps
        return new_mean

    def p_sample(self, xt, mask, t, return_mask=False, return_predictions=False):
        input = torch.cat((xt, mask), dim=1)

        predicted_noise, predicted_mean, predicted_mask = self.model(input, t)
        predicted_mask = predicted_mask.detach()
        mean = self.get_next_step(
            xt, predicted_mean, nn.Sigmoid()(predicted_mask), predicted_noise, t
        )
        if return_predictions:
            return mean, predicted_mean, predicted_mask, predicted_noise
        if return_mask:
            return mean, predicted_mask
        return mean

    def set_noise(self, x0, t, idx, plane_mask=None):
        batch_size = x0.shape[0]
        noise_size = batch_size // 2 if batch_size > 1 else 1
        full_noise = torch.zeros(x0.shape).cuda()
        full_mask = torch.zeros(
            (batch_size, self.mask_chans, x0.shape[-2], x0.shape[-1])
        ).cuda()
        full_mask_c = torch.zeros(
            (batch_size, self.mask_chans, x0.shape[-2], x0.shape[-1])
        ).cuda()
        plane_mask = None if plane_mask is None else plane_mask[:noise_size, :]
        noise, mask, mask_c = self.noise_generator.returnNoise(
            x0[:noise_size, :], t[:noise_size], idx[:noise_size], plane_mask=plane_mask
        )
        full_noise[:noise_size, :] = noise
        full_mask[:noise_size, :] = mask
        full_mask_c[:noise_size, :] = mask_c
        full_mask_c[
            t == self.steps - 1, :
        ] = 0  # The diffusion process always starts with a zero mask
        return full_noise, full_mask, full_mask_c

    def loss(
        self,
        x0,
        noise=None,
        idx=None,
        plane_mask=None,
        save_outputs=False,
        return_loss=False,
    ):
        batch_size = x0.shape[0]

        t = torch.randint(
            1, self.steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        if noise == None:
            noise, mask, mask_c = self.set_noise(
                x0, t, idx, plane_mask=plane_mask
            )  # So that half of the batch size has noise and half doesn't

        xt, noise, mask, mask_c = self.q_sample(x0, mask, t, eps=noise, mask_c=mask_c)
        for b in range(batch_size):
            prob = np.random.rand() > 0.7
            if prob:
                high, low = 20, -20
                angle = np.random.random() * (high - low) + low
                xt[b, :] = torchvision.transforms.functional.rotate(xt[b, :], angle)
                x0[b, :] = torchvision.transforms.functional.rotate(x0[b, :], angle)
                noise[b, :] = torchvision.transforms.functional.rotate(
                    noise[b, :], angle
                )
                mask[b, :] = torchvision.transforms.functional.rotate(mask[b, :], angle)
                mask_c[b, :] = torchvision.transforms.functional.rotate(
                    mask_c[b, :], angle
                )

        input = torch.cat((xt, mask_c), dim=1)

        predicted_noise, predicted_mean, predicted_mask = self.model(input, t)
        beta = gather(self.beta, t)
        noise = beta * noise

        if x0.shape[1] == 4:
            predicted_noise_rgb = predicted_noise[:, :3, :, :].clone()
            noise_rgb = noise[:, :3, :, :].clone()
            predicted_noise_d = predicted_noise[:, 3, :, :].clone()
            noise_d = noise[:, 3, :, :].clone()
            anomaly_loss = (
                F.mse_loss(predicted_noise_rgb, noise_rgb)
                + F.mse_loss(predicted_noise_d, noise_d)
                + self.ssim_loss(predicted_noise_d.unsqueeze(1), noise_d.unsqueeze(1))
            )
        elif x0.shape[1] == 3:
            anomaly_loss = F.mse_loss(predicted_noise, noise)
        else:
            anomaly_loss = F.mse_loss(predicted_noise, noise) + self.ssim_loss(
                predicted_noise, noise
            )

        if x0.shape[1] == 4:
            mask_img = mask[:, :1, :, :]
            predicted_mask_img = predicted_mask[:, :1, :, :]

            mask_loss = 5 * sigmoid_focal_loss(
                predicted_mask_img, mask_img
            ).mean() + F.smooth_l1_loss(nn.Sigmoid()(predicted_mask_img), mask_img)
        else:
            mask_loss = 5 * sigmoid_focal_loss(
                predicted_mask, mask
            ).mean() + F.smooth_l1_loss(nn.Sigmoid()(predicted_mask), mask)
        s_loss = 0
        if x0.shape[1] == 4:
            predicted_mean_img = predicted_mean[:, :3, :, :].clone()
            predicted_mean_d = predicted_mean[:, 3:, :, :].clone()

            x0_img = x0[:, :3, :, :].clone()
            x0_d = x0[:, 3:, :, :].clone()

            rgb_ssim_loss = self.ssim_loss(predicted_mean_img, x0_img)
            rgb_l1_loss = F.l1_loss(predicted_mean_img, x0_img)

            d_ssim_loss = self.ssim_loss(predicted_mean_d, x0_d)
            d_l1_loss = F.l1_loss(predicted_mean_d, x0_d)

            s_loss = rgb_ssim_loss + d_ssim_loss
            l1_loss = rgb_l1_loss + d_l1_loss
        else:
            s_loss = self.ssim_loss(predicted_mean, x0)
            l1_loss = F.l1_loss(predicted_mean, x0)

        normal_loss = s_loss + l1_loss

        next_predicted = self.get_next_step(
            xt, predicted_mean, nn.Sigmoid()(predicted_mask), predicted_noise, t
        )
        next_actual = self.get_next_step(xt, x0, mask, noise, t)
        img_loss = F.mse_loss(next_predicted, next_actual)

        if return_loss:
            return (
                anomaly_loss + mask_loss + normal_loss + img_loss,
                anomaly_loss,
                mask_loss,
                normal_loss,
                img_loss,
            )
        return anomaly_loss + mask_loss + normal_loss + img_loss
