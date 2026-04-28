# 点群法で1つのゾーンプレートを表示するスクリプト

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.data
import tqdm
from constants import ClassicalConstants
from monoq import monopolar_fixed_point, random_hologram, sum_holograms
from pointcloud import (
    create_circle,
    create_cube,
    create_depth_line,
    create_rectangle_points,
    create_sin_wave,
    create_single_point,
    generate_hologram,
    show,
)
from reconst_hologram import fresnel_fft, response


def load_bunny_pointcloud() -> open3d.geometry.PointCloud:
    bunny_path = open3d.data.BunnyMesh().path
    point_cloud = o3d.io.read_point_cloud(bunny_path)
    return point_cloud


def downsampling(
    point_cloud: open3d.geometry.PointCloud, every_k_points: int = 10
) -> open3d.geometry.PointCloud:
    points = point_cloud.uniform_down_sample(every_k_points=every_k_points)
    return points


def show_pointcloud(points: np.ndarray) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=4)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Point Cloud")
    ax.set_box_aspect(np.ptp(points, axis=0))
    plt.tight_layout()
    plt.show()


def transform_points_to_plate(
    points: np.ndarray,
    constants: ClassicalConstants,
    show_projection: bool = True,
) -> np.ndarray:
    # 3d bunny を z 軸方向から見た 2d 点群に変換する。
    # generate_hologram() は x, y を画素座標、z を距離[m]として扱う。
    xy = points[:, :2]
    xy_min = xy.min(axis=0)
    xy_size = np.ptp(xy, axis=0)
    xy_size[xy_size == 0.0] = 1.0

    scale = (
        min(
            (constants.X - 1) / xy_size[0],
            (constants.Y - 1) / xy_size[1],
        )
        * 0.7
    )  # X,Yいっぱいにならないように小さく
    scaled_xy = (xy - xy_min) * scale
    offset = np.array(
        [
            (constants.X - 1 - scaled_xy[:, 0].max()) / 2.0,
            (constants.Y - 1 - scaled_xy[:, 1].max()) / 2.0,
        ]
    )
    image_xy = scaled_xy + offset

    points_2d = np.column_stack(
        (
            image_xy[:, 0],
            image_xy[:, 1],
            points[:, 2],
        )
    )

    if show_projection:
        projection = np.zeros((constants.Y, constants.X), dtype=np.float64)
        xi = np.rint(points_2d[:, 0]).astype(np.int32)
        yi = np.rint(points_2d[:, 1]).astype(np.int32)
        valid = (
            (0 <= xi) & (xi < constants.X) & (0 <= yi) & (yi < constants.Y)
        )
        projection[yi[valid], xi[valid]] = 1.0

        fig, ax = plt.subplots()
        ax.imshow(projection, cmap="gray", origin="lower")
        ax.set_title("Bunny Projection")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.tight_layout()
        plt.show()

    return points_2d


def adapt_z_to_depth(
    points: np.ndarray,
    source_points: np.ndarray,
    constants: ClassicalConstants,
    depth_range: float = 30e-3,
    show_depth: bool = True,
) -> np.ndarray:
    # bunnyの深度をconstants.dに代入する
    if len(points) != len(source_points):
        raise ValueError("points and source_points must have the same length")

    z = source_points[:, 2].astype(np.float64)
    z_min = z.min()
    z_size = np.ptp(z)

    if z_size == 0.0:
        depth = np.full(len(points), constants.d, dtype=np.float64)
    else:
        z_norm = (z - z_min) / z_size
        depth = constants.d + (z_norm - 0.5) * depth_range

    points_with_depth = points.copy()
    points_with_depth[:, 2] = depth

    if show_depth:
        fig, ax = plt.subplots()
        scatter = ax.scatter(
            points_with_depth[:, 0],
            points_with_depth[:, 1],
            c=points_with_depth[:, 2],
            s=4,
            cmap="viridis",
        )
        fig.colorbar(scatter, ax=ax, label="Depth [m]")
        ax.set_title("Bunny Depth")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.show()

    return points_with_depth


def main():
    start = time.time()
    constants = ClassicalConstants()

    # points
    points = create_single_point(constants)
    # points = create_rectangle_points(constants)
    # points = create_sin_wave(constants)
    # points = create_circle(constants)

    # # bunny
    # point_cloud = load_bunny_pointcloud()
    # print("Loading data completed!")
    # point_cloud = downsampling(point_cloud, every_k_points=10)
    # print("Downsampling completed!")
    # points = np.asarray(point_cloud.points)

    # cube
    # points = create_cube(constants, width=constants.X // 8, step=15)

    # depth_only
    # points = create_depth_line(depth_n=10000, constants=constants)

    show_pointcloud(points)

    out = []
    cgh = False  # True
    if cgh:
        # cghホログラム、再構成
        hologram = generate_hologram(points, constants)  # ok
        recon = fresnel_fft(hologram.astype(np.complex128), constants)
        recon_intensity = np.abs(recon) ** 2
        out = [hologram, recon_intensity]
    else:
        # # monopolarホログラム
        # points = transform_points_to_plate(points, constants)  # 2dにする
        hologram = monopolar_fixed_point(points, constants, binary=True)
        holo_ratio = sum_holograms(hologram, len(points))  # /物体点
        holo_rand = random_hologram(
            holo_ratio=holo_ratio, constants=constants
        )

        # # monopolar再構成
        recon_holo = fresnel_fft(holo_ratio.astype(np.complex128), constants)
        holo_reconst_intensity = np.abs(recon_holo) ** 2
        recon_rand = fresnel_fft(holo_rand.astype(np.complex128), constants)
        rand_reconst_intensity = np.abs(recon_rand) ** 2

        out = [
            holo_ratio,
            holo_reconst_intensity,
            holo_rand,
            rand_reconst_intensity,
        ]

    print("CGH Calculation completed!")

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))

    print("Preparing for display...")
    show(
        out,
        x=constants.X,
        y=constants.Y,
        binary=True,
        target=0,
        dir=Path("./results/images/3d/bunny_reconst.png"),
    )


if __name__ == "__main__":
    main()

# TODO - binaryにbunnyをいれてみる
# TODO - 3d点群対応
