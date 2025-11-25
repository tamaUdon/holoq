import math
import open3d as o3d
import open3d.data
import numpy as np
import matplotlib.pyplot as plt
import time

X = 1920  # 200万画素の場合
Y = 1080
PXL = X * Y
λ = 500  # 波長
k = 2 * math.pi / λ


def load_pointcloud() -> np.ndarray:
    bunny_path = open3d.data.BunnyMesh().path
    point_cloud = o3d.io.read_point_cloud(bunny_path)
    points = np.asarray(point_cloud.points)
    # print(points.shape)
    # print(points.size)
    return points


def calculate_hologram(data: np.ndarray) -> tuple:
    Ix_holo = []
    Iy_holo = []
    for y_i in range(Y):
        for x_i in range(X):
            for d in data:  # WARNING - 各次元が同じsizeだと仮定している
                x_j = d[0]
                y_j = d[1]
                z_j = d[2]

                x_p = ((PXL * x_i) - x_j) ** 2
                y_p = ((PXL * y_i) - y_j) ** 2
                z_p = z_j**2

                r = math.sqrt(((x_p**2) + (y_p**2) + (z_p**2)))
                I_tmp = (1 / r) * math.cos(k * r)
                Ix_holo.append(I_tmp + x_p)
                Iy_holo.append(I_tmp + y_p)
    return (Ix_holo, Iy_holo)


def show_hologram(Ix_holo: list, Iy_holo: list):
    plt.plot(Ix_holo, Iy_holo, label="holography")
    plt.legend()
    plt.show()


def main():
    start = time.time()

    points = load_pointcloud()
    (Ix_holo, Iy_holo) = calculate_hologram(points)
    end = time.time()

    print(print("Cal time:{}".format(end - start)))
    show_hologram(Ix_holo, Iy_holo)


if __name__ == "__main__":
    main()
