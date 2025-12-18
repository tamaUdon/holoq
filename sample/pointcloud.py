# 点群法で1つのゾーンプレートを表示するスクリプト

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
    """
    Constants の Docstring

    :param DEBUG: デバッグON/OFF
    :type DEBUG: Bool
    :param X,Y: 画素数
    :type X,Y: int
    :param λ: 波長
    :type λ: int (nm) # TODO - [int]にする
    :param k: 波数 (2pi/λ)
    :type k: int
    :param pp: 画素ピッチ
    :type pp: int
    :param d: ホログラムと物体間の距離
    :type d: int

    :return: bipolarホログラムの計算結果
    :rtype: np.ndarray
    """

    DEBUG = True
    X = 500  # 画素X方向
    Y = X
    λ = 500e-9  # 波長[nm]
    k = 2 * math.pi / λ
    pp = 10e-6  # 画素ピッチ[μm]
    d = 1000e-3  # 物体までの距離[mm]


def create_single_point(constants: Constants) -> np.ndarray:
    """
    create_single_point の Docstring

    - X*Yの中心に物体点 (1点) がある想定

    :param constants: 定数クラスのオブジェクト
    :type constants: Constants
    :return: デバッグ用の物体点 (1点)
    :rtype: np.ndarray
    """
    x0 = constants.X / 2
    y0 = constants.Y / 2
    z0 = constants.d  # 物体点までの距離

    return np.array([[x0, y0, z0]], dtype=float)


def load_bunny_pointcloud() -> open3d.geometry.PointCloud:
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


def calculate_zoneplate(data: np.ndarray, constants: Constants) -> np.ndarray:
    I_holography = np.zeros((constants.Y, constants.X))

    print("Calculating CGH...")
    for y_i in tqdm.tqdm(range(constants.Y)):
        for x_i in range(constants.X):
            for dt in data:
                x_j = dt[0]
                y_j = dt[1]
                z_j = dt[2]

                x_p = ((x_i) * constants.pp - x_j * constants.pp) ** 2
                y_p = ((y_i) * constants.pp - y_j * constants.pp) ** 2
                z_p = z_j**2

                r = math.sqrt((x_p + y_p + z_p))
                I_tmp = (1 / r) * math.cos(constants.k * r)
                I_holography[y_i, x_i] = I_tmp
    print("CGH Calculation completed!")
    return I_holography


def show_graph(I_holography: np.ndarray) -> None:
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
    points = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])

    if constants.DEBUG:
        points = create_single_point(constants)  # TODO - multi-pointsに
    else:
        point_cloud = load_bunny_pointcloud()
        point_cloud = downsampling(point_cloud, every_k_points=1000)
        points = np.asarray(point_cloud.points)

    plate = calculate_zoneplate(points, constants)  # TODO - ここで足し合わせ np.add?

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))

    show_graph(plate)


if __name__ == "__main__":
    main()

# TODO
# 1. create_single_point を multi_points にする

# TODO
# 1. コマンドライン引数を受け取れるようにする
# 2. Constantsに引数データを入れる
# 3. 複数波長を受け取れるようにλを[]にする
