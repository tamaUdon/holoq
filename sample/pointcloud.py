import math
import time
import tqdm
import open3d.data
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import dataclasses


@dataclasses.dataclass
class Constants:
    X = 192  # 200万画素の場合
    Y = 108
    PXL = X * Y
    λ = 500  # 波長
    k = 2 * math.pi / λ


def load_pointcloud() -> open3d.geometry.PointCloud:
    bunny_path = open3d.data.BunnyMesh().path
    point_cloud = o3d.io.read_point_cloud(bunny_path)
    print("Loading data completed!")
    return point_cloud


def downsampling(
    point_cloud: open3d.geometry.PointCloud, every_k_points: int = 10
) -> open3d.geometry.PointCloud:
    points = point_cloud.uniform_down_sample(every_k_points=every_k_points)
    print("Downsampling completed!")
    return points


def calculate_holography(data: np.ndarray, constants: Constants) -> np.ndarray:
    I_holography = np.zeros((constants.Y, constants.X))

    print("Calculating CGH...")
    for y_i in tqdm.tqdm(range(constants.Y)):
        for x_i in range(constants.X):
            for d in data:
                x_j = d[0]
                y_j = d[1]
                z_j = d[2]

                x_p = ((constants.PXL * x_i) - x_j) ** 2
                y_p = ((constants.PXL * y_i) - y_j) ** 2
                z_p = z_j**2

                r = math.sqrt((x_p + y_p + z_p))
                I_tmp = (1 / r) * math.cos(constants.k * r)
                I_holography[y_i, x_i] = I_tmp
    print("CGH Calculation completed!")
    return I_holography


def show_hologram(I_holography: np.ndarray) -> None:
    print("Preparing for display...")

    fig, ax = plt.subplots()
    CS = ax.contourf(range(Constants.X), range(Constants.Y), I_holography)
    fig.colorbar(CS)
    fig.set_label("holography")
    plt.legend()
    plt.show()


def main():
    start = time.time()
    print("Preparing for CGH...")

    constants = Constants()
    point_cloud = load_pointcloud()
    point_cloud = downsampling(point_cloud, every_k_points=1000)
    points = np.asarray(point_cloud.points)
    holography = calculate_holography(points, constants)

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))

    show_hologram(holography)


if __name__ == "__main__":
    main()

# TODO
# 1. コマンドライン引数を受け取れるようにする
# 2. Constantsに引数データを入れる
# 3. 複数波長を受け取れるようにλを[]にする
