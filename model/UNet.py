import math

import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncodingPermute2D


class SkipConnection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.skip_con = nn.Identity()

    def forward(self, x):
        return self.skip_con(x)


class SegmentationHead(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        x = self.act(self.norm(x))
        x = self.conv(x)
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.num_of_channels = n_channels
        self.linear_1 = nn.Linear(self.num_of_channels // 4, self.num_of_channels)
        self.linear_2 = nn.Linear(self.num_of_channels, self.num_of_channels)
        self.act = nn.SiLU()

    def forward(self, t):
        half_dim = self.num_of_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        emb = self.act(self.linear_1(emb))
        emb = self.linear_2(emb)

        return emb


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_channels,
        n_groups=32,
        groups=1,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), groups=groups
        )

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            groups=groups,
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=(1, 1), groups=groups
            )
        else:
            self.shortcut = nn.Identity()
        self.time_emb = nn.Linear(time_channels, out_channels)

    def forward(self, x, t):
        h = self.conv1(self.act1(self.norm1(x)))
        t = self.time_emb(t)[:, :, None, None]
        h += t
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, n_channels, n_heads=1, d_k=None, n_groups=32):
        super().__init__()
        if d_k is None:
            d_k = n_channels // n_heads
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k**-0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t=None):
        _ = t
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)

        res += x

        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        return res


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool,
        groups: int = 1,
    ):
        super().__init__()

        self.groups = groups
        self.has_attn = has_attn
        self.res = ResidualBlock(
            in_channels, out_channels, time_channels, groups=groups
        )
        if has_attn:
            if self.groups > 1:
                self.attn = AttentionBlock(out_channels // 2)
                self.attn_d = AttentionBlock(out_channels // 2)
            else:
                self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        if self.has_attn and self.groups > 1:
            c = x.shape[1]
            x_rgb = x[:, : c // 2, :, :]
            x_d = x[:, c // 2 :, :, :]
            x_rgb = self.attn(x_rgb)
            x_d = self.attn_d(x_d)
            x = torch.cat([x_rgb, x_d], dim=1)
        else:
            x = self.attn(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool,
        groups: int = 1,
    ):
        super().__init__()
        add = out_channels
        self.groups = groups
        self.has_attn = has_attn
        self.res = ResidualBlock(
            in_channels + add, out_channels, time_channels, groups=groups
        )
        if has_attn:
            if self.groups > 1:
                self.attn = AttentionBlock(out_channels // 2)
                self.attn_d = AttentionBlock(out_channels // 2)
            else:
                self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        if self.groups > 1:
            c = x.shape[1]
            x_prev_rgb = x[:, : c // 4, :, :]
            x_prev_d = x[:, c // 4 : c // 2, :, :]
            x_rgb = x[:, c // 2 : 3 * c // 4, :, :]
            x_d = x[:, 3 * c // 4 :, :, :]
            x = torch.cat([x_prev_rgb, x_rgb, x_prev_d, x_d], dim=1)
        x = self.res(x, t)
        if self.has_attn and self.groups > 1:
            c = x.shape[1]
            x_rgb = x[:, : c // 2, :, :]
            x_d = x[:, c // 2 :, :, :]
            x_rgb = self.attn(x_rgb)
            x_d = self.attn_d(x_d)
            x = torch.cat([x_rgb, x_d], dim=1)
        else:
            x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, time_channels: int, groups: int = 1):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels, groups=groups)

        self.groups = groups
        if self.groups > 1:
            self.attn = AttentionBlock(n_channels // 2)
            self.attn_d = AttentionBlock(n_channels // 2)
        else:
            self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels, groups=groups)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        if self.groups > 1:
            c = x.shape[1]
            x_rgb = x[:, : c // 2, :, :]
            x_d = x[:, c // 2 :, :, :]
            x_rgb = self.attn(x_rgb)
            x_d = self.attn_d(x_d)
            x = torch.cat([x_rgb, x_d], dim=1)
        else:
            x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    def __init__(self, n_channels, groups=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            n_channels, n_channels, (4, 4), (2, 2), (1, 1), groups=groups
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, n_channels, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            n_channels, n_channels, (3, 3), (2, 2), (1, 1), groups=groups
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, parameters: dict):
        super().__init__()

        n_channels = parameters["n_channels"]
        ch_mults = parameters["ch_mults"]
        is_attn = parameters["is_attn"]
        use_groups = parameters["use_groups"]
        n_blocks = parameters["n_blocks"]
        image_channels = int(parameters["image_channels"])
        self.is_rgbd = parameters["is_rgbd"]
        self.groups = list(
            [2 if use_groups[i] and self.is_rgbd else 1 for i in range(len(use_groups))]
        )
        self.start_img_channels = image_channels
        self.n_layers = len(ch_mults)

        n_resolutions = len(ch_mults)

        image_channels = image_channels + 1  # Mask
        image_channels = image_channels + 1  # PE

        self.pos_emb = PositionalEncodingPermute2D(1)

        if self.groups[0] == 2:
            self.image_proj_rgb = nn.Conv2d(
                image_channels - 1, n_channels // 2, kernel_size=(3, 3), padding=(1, 1)
            )
            self.image_proj_d = nn.Conv2d(
                image_channels - 3, n_channels // 2, kernel_size=(3, 3), padding=(1, 1)
            )
        else:
            self.image_proj = nn.Conv2d(
                image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1)
            )

        self.time_emb = TimeEmbedding(n_channels * 4)

        down = []
        skip_con = []
        self.cond_embs = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels,
                        out_channels,
                        n_channels * 4,
                        is_attn[i],
                        groups=self.groups[i],
                    )
                )
                skip_con.append(SkipConnection())
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels, groups=self.groups[i]))
                skip_con.append(SkipConnection())

        self.skip_con = nn.ModuleList(skip_con)
        self.down = nn.ModuleList(down)
        mid_group = 2 if self.groups[-1] > 1 else 1
        self.middle = MiddleBlock(out_channels, n_channels * 4, groups=mid_group)

        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels,
                        out_channels,
                        n_channels * 4,
                        is_attn[i],
                        groups=self.groups[i],
                    )
                )
            out_channels = in_channels // ch_mults[i]
            up.append(
                UpBlock(
                    in_channels,
                    out_channels,
                    n_channels * 4,
                    is_attn[i],
                    groups=self.groups[i],
                )
            )
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels, groups=self.groups[i]))
        self.up = nn.ModuleList(up)

        self.norm = nn.GroupNorm(8, in_channels)
        self.act = nn.SiLU()

        image_channels = image_channels - 2

        self.norm = nn.GroupNorm(8, in_channels)
        self.act = nn.SiLU()

        if self.groups[0] == 2:
            self.final = nn.Conv2d(
                in_channels // 2, 3, kernel_size=(3, 3), padding=(1, 1)
            )
            self.final_mean = nn.Conv2d(
                in_channels // 2, 3, kernel_size=(3, 3), padding=(1, 1)
            )
            self.final_d = nn.Conv2d(
                in_channels // 2, 1, kernel_size=(3, 3), padding=(1, 1)
            )
            self.final_mean_d = nn.Conv2d(
                in_channels // 2, 1, kernel_size=(3, 3), padding=(1, 1)
            )
        else:
            self.final = nn.Conv2d(
                in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1)
            )
            self.final_mean = nn.Conv2d(
                in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1)
            )
        self.final_seg = SegmentationHead(in_channels)

    def prepare_start(self, x, t):
        t = self.time_emb(t)

        b, c, x_shape, y_shape = x.shape
        z = torch.zeros((b, 1, x_shape, y_shape)).cuda()
        pos = self.pos_emb(z)
        if self.groups[0] == 2:
            x_rgb = x[:, :3, :, :]
            x_d = x[:, 3:4, :, :]
            m_rgb = x[:, 4:, :, :]
            m_d = x[:, 4:, :, :]
            x_rgb = torch.cat([x_rgb, m_rgb, pos], dim=1)
            x_d = torch.cat([x_d, m_d, pos], dim=1)
            x_rgb = self.image_proj_rgb(x_rgb)
            x_d = self.image_proj_d(x_d)
            x = torch.cat([x_rgb, x_d], dim=1)
        else:
            x = torch.cat((x, pos), axis=1)
            x = self.image_proj(x)

        return x, t

    def encode(self, x, t):
        h = [x]
        for m, sc in zip(self.down, self.skip_con):
            x = m(x, t)
            h.append(sc(x))
        return x, h

    def middle_block(self, x, t):
        x = self.middle(x, t)
        return x

    def decode(self, x, h, t):
        for m in self.up:
            if not isinstance(m, Upsample):
                s = h.pop()
                x = torch.cat((x, s), dim=1)
            x = m(x, t)
        return x

    def final_output(self, x):
        c = x.shape[1]
        f = self.act(self.norm(x))
        if self.is_rgbd:
            mask = self.final_seg(x)
            if self.groups[0] == 2:
                f_d = self.final_d(f[:, c // 2 :, :, :])
                mean_d = self.final_mean_d(x[:, c // 2 :, :, :])
                f = self.final(f[:, : c // 2, :, :])
                mean = self.final_mean(x[:, : c // 2, :, :])
                f = torch.cat([f, f_d], dim=1)
                mean = torch.cat([mean, mean_d], dim=1)
            else:
                f = self.final(f)
                mean = self.final_mean(x)
            return f, mean, mask
        f = self.final(f)
        return f, self.final_mean(x), self.final_seg(x)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x, t = self.prepare_start(x, t)
        x, h = self.encode(x, t)
        x = self.middle_block(x, t)
        x = self.decode(x, h, t)
        return self.final_output(x)
