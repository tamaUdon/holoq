# 点群法で1つのゾーンプレートを表示するスクリプト

import math
import time
import tqdm
import open3d.data
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pointcloud import generate_hologram
from constants import ClassicalConstants


def load_bunny_pointcloud() -> open3d.geometry.PointCloud:
    bunny_path = open3d.data.BunnyMesh().path
    point_cloud = o3d.io.read_point_cloud(bunny_path)
    return point_cloud


def downsampling(
    point_cloud: open3d.geometry.PointCloud, every_k_points: int = 10
) -> open3d.geometry.PointCloud:
    points = point_cloud.uniform_down_sample(every_k_points=every_k_points)
    return points


def save(
    holography: np.ndarray, constants: ClassicalConstants, fname: str
) -> None:
    fig, ax = plt.subplots()
    color = ax.contourf(range(constants.X), range(constants.Y), holography)
    fig.colorbar(color)
    fig.set_label("holography")
    fig.savefig(fname=fname)


def save_point_cloud(point_cloud: np.ndarray, fname: str):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)
    fig.set_label("3d point cloud")
    fig.savefig(fname=fname)


def main():
    start = time.time()
    constants = ClassicalConstants()

    point_cloud = load_bunny_pointcloud()
    print("Loading data completed!")

    point_cloud = downsampling(point_cloud, every_k_points=10)
    point_array = np.asarray(point_cloud.points)
    save_point_cloud(point_array, "results/images/3d/pc_bunny.png")
    print("Downsampling completed!")

    points = np.asarray(point_cloud.points)
    plate = generate_hologram(points, constants)
    print("CGH Calculation completed!")

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))

    print("Preparing for display...")
    save(plate, constants, "results/images/3d/bunny.png")


if __name__ == "__main__":
    main()
