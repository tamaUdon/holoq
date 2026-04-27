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
from pointcloud import generate_hologram, show
from reconst_hologram import fresnel_fft


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


def main():
    start = time.time()
    constants = ClassicalConstants()

    point_cloud = load_bunny_pointcloud()
    print("Loading data completed!")

    point_cloud = downsampling(point_cloud, every_k_points=10)
    print("Downsampling completed!")

    points = np.asarray(point_cloud.points)
    show_pointcloud(points)
    hologram = generate_hologram(points, constants)
    reconst = fresnel_fft(hologram.astype(np.complex128), constants)

    # # monopolarホログラム
    # hologram = monopolar_fixed_point(points, constants, binary=True)
    # holo_ratio = sum_holograms(hologram, len(points))  # /物体点

    # # QGHに適用 (random)
    # hologram_rand = random_hologram(
    #     holo_ratio=holo_ratio, constants=constants
    # )

    # # 再構成して出力を見る
    # reconst = fresnel_fft(hologram_rand.astype(np.complex128), constants)
    # reconst_intensity = np.abs(reconst) ** 2
    print("CGH Calculation completed!")

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))

    print("Preparing for display...")
    show(
        # [holo_ratio, hologram_rand, reconst_intensity],
        [hologram, reconst],
        x=constants.X,
        y=constants.Y,
        binary=True,
        target=0,
        dir=Path("./results/images/3d/bunny_reconst.png"),
    )


if __name__ == "__main__":
    main()
