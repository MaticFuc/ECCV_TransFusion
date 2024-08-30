import cv2
import imageio.v3 as iio
import numpy as np
import torch
import yaml


def fill_depth_map(depth_image, iterations=2):
    dimg = depth_image
    for i in range(iterations):
        zero_mask = np.where(dimg == 0, np.ones_like(dimg), np.zeros_like(dimg))
        dimg_tensor = torch.from_numpy(dimg)
        h, w = dimg_tensor.shape
        dimg_tensor = dimg_tensor.reshape((1, 1, h, w))  # use only depth
        dimg_t = torch.nn.functional.unfold(
            dimg_tensor, 3, dilation=1, padding=1, stride=1
        )  # B, 1x3x3, L -> L=HW
        dimg_t_nonzero_sum = torch.sum(
            torch.where(dimg_t > 0, torch.ones_like(dimg_t), torch.zeros_like(dimg_t)),
            dim=1,
            keepdim=True,
        )
        dimg_t_sum = torch.sum(dimg_t, dim=1, keepdim=True)  # B, 1, L
        dimg_t_filtered = dimg_t_sum / (dimg_t_nonzero_sum + 1e-12)
        dimg_out = torch.nn.functional.fold(
            dimg_t_filtered, dimg.shape[:2], 1, dilation=1, padding=0, stride=1
        )  # B, 1, H, W
        dimg = dimg_out.numpy()[0, 0, :, :] * zero_mask + (1.0 - zero_mask) * dimg
    return dimg


def fill_plane_mask(plane_mask):
    kernel = np.ones((3, 3), np.uint8)
    plane_mask[:, :, 0] = cv2.morphologyEx(
        plane_mask, cv2.MORPH_CLOSE, kernel, iterations=3
    )
    return plane_mask


def get_plane_mask(depth_image):
    h, w, c = depth_image.shape
    points = np.reshape(depth_image, (h * w, c))
    p1 = np.sum(depth_image[:3, :3, :], axis=(0, 1)) / np.sum(
        depth_image[:3, :3, 2] != 0
    )
    p2 = np.sum(depth_image[:3, -3:, :], axis=(0, 1)) / np.sum(
        depth_image[:3, -3:, 2] != 0
    )
    p3 = np.sum(depth_image[-3:, :3, :], axis=(0, 1)) / np.sum(
        depth_image[-3:, :3, 2] != 0
    )
    p4 = np.sum(depth_image[-3:, -3:, :], axis=(0, 1)) / np.sum(
        depth_image[-3:, -3:, 2] != 0
    )
    plane = get_plane_from_points(p1, p2, p3)
    point_distance = get_distance_to_plane(points, np.array(plane))
    points_mask = np.where(
        point_distance > 0.005,
        np.ones_like(point_distance),
        np.zeros_like(point_distance),
    )
    plane_mask = np.reshape(points_mask, (h, w, 1))
    return plane_mask


def get_plane_from_points(p1, p2, p3):
    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    # print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))
    # plane equation: ax + by + cz - d = 0
    return a, b, c, d


def get_distance_to_plane(points, plane):
    # points - np array in form [N, 3]
    # plane in the form of  [a,b,c,d]
    plane_rs = np.expand_dims(plane, 0)
    dist = (
        np.abs(np.sum(points * plane_rs[:, :-1], axis=1) - plane[-1])
        / np.sum(plane[:-1] ** 2) ** 0.5
    )
    return dist


def get_plane_points(p, plane):
    # p -- N x 3
    a, b, c, d = plane
    normal = np.zeros(1, 3)
    normal[0, :] = np.array([a, b, c])
    c = (d - np.sum(p * normal, axis=1)) / np.sum(normal**2)
    out_points = p + c * normal
    return out_points


def get_plane_mask_eyecandy(depth_image, thr=0.005):
    h, w, c = depth_image.shape
    points = np.reshape(depth_image, (h * w, c))
    p1 = depth_image[0, w // 2, :]
    p2 = depth_image[h - 1, 0, :]
    p3 = depth_image[h - 1, w - 1, :]
    plane = get_plane_from_points(p1, p2, p3)
    point_distance = get_distance_to_plane(points, np.array(plane))
    points_mask = np.where(
        point_distance > thr,
        np.ones_like(point_distance),
        np.zeros_like(point_distance),
    )
    plane_mask = np.reshape(points_mask, (h, w, 1))
    return plane_mask


def load_and_convert_depth(depth_img, info_depth):
    with open(info_depth) as f:
        data = yaml.safe_load(f)
    mind, maxd = data["normalization"]["min"], data["normalization"]["max"]
    dimg = iio.imread(depth_img)
    dimg = dimg.astype(np.float32)
    dimg = dimg / 65535.0 * (maxd - mind) + mind
    return dimg


def depth_to_pointcloud(depth_img, info_depth, pose_txt):
    # input depth map (in meters) --- cfr previous section
    focal_length = 711.11
    depth_mt = load_and_convert_depth(depth_img, info_depth)
    # input pose
    pose = np.loadtxt(pose_txt)
    # camera intrinsics
    height, width = depth_mt.shape[:2]
    intrinsics_4x4 = np.array(
        [
            [focal_length, 0, width / 2, 0],
            [0, focal_length, height / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    # build the camera projection matrix
    camera_proj = intrinsics_4x4 @ pose
    # build the (u, v, 1, 1/depth) vectors (non optimized version)
    # camera_vectors = np.zeros((width * height, 4))
    # camera_vectors = np.zeros((height, width, 4))
    coords_x = np.expand_dims(
        np.tile(np.arange(0, width), (height, 1)), axis=2
    )  # H x W
    coords_y = np.expand_dims(
        np.tile(np.arange(0, height), (width, 1)).T, axis=2
    )  # H x W
    ones_t = np.ones((height, width, 1))
    depth_inv = np.expand_dims(1.0 / depth_mt, axis=2)
    camera_vectors = np.concatenate(
        (coords_x, coords_y, ones_t, depth_inv), axis=2
    )  # h x w x 4
    # invert and apply to each 4-vector -- 3 x 4 X 4 x h x w -- 3 x h x w
    # hom_3d_pts= np.linalg.inv(camera_proj) @ camera_vectors.transpose((2,0,1))
    hom_3d_pts = np.einsum(
        "ij,jlm->ilm", np.linalg.inv(camera_proj), camera_vectors.transpose((2, 0, 1))
    )
    # remove the homogeneous coordinate
    # hw,1 x hw,3
    # pcd = depth_mt.reshape(-1, 1) * hom_3d_pts.T
    pcd = depth_mt.reshape(height, width, 1) * hom_3d_pts.transpose((1, 2, 0))  # h w 3
    # return pcd[:, :3]
    return pcd
