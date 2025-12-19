# 点群法で1つのゾーンプレートを表示するスクリプト

import math
import time
import tqdm
import open3d.data
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import dataclasses


@dataclasses.dataclass(frozen=True)
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
    X = 100  # 画素X方向
    Y = X
    λ = 500e-9  # 波長[nm]
    pp = 10e-6  # 画素ピッチ[μm]
    d = 10e-3  # 物体までの距離[mm]

    @property
    def k(self) -> float:
        return 2 * math.pi / self.λ


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


def create_four_points(constants: Constants) -> np.ndarray:
    """
    create_rect_points 4点ゾーンプレートの点群を作成する関数

    :param constants: 定数クラスのオブジェクト
    :type constants: Constants
    """

    center = np.array([constants.X / 2, constants.Y / 2, constants.d])
    half = np.array([constants.X / 4, constants.Y / 4, 0.0])

    signs = np.array(
        [
            [1, 1, 0],
            [-1, 1, 0],
            [-1, -1, 0],
            [1, -1, 0],
        ]
    )

    return center + signs * half


def create_rectangle_points(constants: Constants) -> np.ndarray:
    """
    create_rectangle_points 四角形点群を作成

    :param constants: 定数クラスのオブジェクト
    :type constants: Constants
    """

    x_size = constants.X // 2
    dx = np.array([constants.X / 4, 0.0, 0.0])
    dy = np.array([0.0, constants.Y / 4, 0.0])

    x_line = np.array([[x, constants.Y // 2, constants.d] for x in range(x_size)])
    y_line = np.array([[constants.X // 2, y, constants.d] for y in range(x_size)])

    # lineをスライドさせて四角形を作る TODO - numpy関数を使う
    top = x_line + dy + dx
    bottom = x_line - dy + dx

    left = y_line + dy + dx
    right = y_line + dy - dx

    rectangle = np.concatenate((top, bottom, left, right))
    return rectangle


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


def generate_hologram(points: np.ndarray, constants: Constants) -> np.ndarray:
    print("Calculating Hologram...")
    x = np.arange(constants.X, dtype=np.float64) * constants.pp
    y = np.arange(constants.Y, dtype=np.float64) * constants.pp
    xx, yy = np.meshgrid(x, y)
    hologram = np.zeros((constants.Y, constants.X), dtype=np.float64)

    for xj, yj, zj in tqdm.tqdm(points):
        dx = xx - xj * constants.pp
        dy = yy - yj * constants.pp
        r = np.sqrt(dx * dx + dy * dy + zj * zj)
        hologram += np.cos(constants.k * r) / r
    print("CGH Calculation completed!")
    return hologram


def show(I_holography: np.ndarray) -> None:
    print("Preparing for display...")

    fig, ax = plt.subplots()
    CS = ax.contourf(range(Constants.X), range(Constants.Y), I_holography)
    fig.colorbar(CS)
    fig.set_label("holography")
    plt.legend()
    plt.show()


def main():
    start = time.time()
    constants = Constants()
    points = np.array([[0, 0, 0]])

    if constants.DEBUG:
        # points = create_four_points(constants) # 4点
        points = create_rectangle_points(constants)  # 四角形 # TODO - 分岐
    else:
        point_cloud = load_bunny_pointcloud()
        point_cloud = downsampling(point_cloud, every_k_points=1000)
        points = np.asarray(point_cloud.points)
    plate = generate_hologram(points, constants)

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))
    show(plate)


if __name__ == "__main__":
    main()

# TODO
# 1. コマンドライン引数を受け取れるようにする
# 2. Constantsに引数データを入れる
# 3. 複数波長を受け取れるようにλを[]にする
