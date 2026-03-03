# 点群法で1つのゾーンプレートを表示するスクリプト

import math
import time
import tqdm
import open3d.data
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pointcloud import generate_hologram
from constants import Constants


def load_bunny_pointcloud() -> open3d.geometry.PointCloud:
    bunny_path = open3d.data.BunnyMesh().path
    point_cloud = o3d.io.read_point_cloud(bunny_path)
    return point_cloud


def downsampling(
    point_cloud: open3d.geometry.PointCloud, every_k_points: int = 10
) -> open3d.geometry.PointCloud:
    points = point_cloud.uniform_down_sample(every_k_points=every_k_points)
    return points


def show(holography: np.ndarray) -> None:
    fig, ax = plt.subplots()
    color = ax.contourf(range(Constants.X), range(Constants.Y), holography)
    fig.colorbar(color)
    fig.set_label("holography")
    plt.legend()
    plt.show()


def main():
    start = time.time()
    constants = Constants()

    point_cloud = load_bunny_pointcloud()
    print("Loading data completed!")

    point_cloud = downsampling(point_cloud, every_k_points=1000)
    print("Downsampling completed!")

    points = np.asarray(point_cloud.points)
    plate = generate_hologram(points, constants)
    print("CGH Calculation completed!")

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))

    print("Preparing for display...")
    show(plate)


if __name__ == "__main__":
    main()
