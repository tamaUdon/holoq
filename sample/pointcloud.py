# 点群法でホログラムを表示するスクリプト

import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from constants import ClassicalConstants, QuantumConstants


def create_single_point(constants: ClassicalConstants) -> np.ndarray:
    """
    画面中心に 1 点だけ配置した点群を生成する。

    Args:
        constants: 点群生成に使用する定数。

    Returns:
        形状 `(1, 3)` の点群配列。
    """
    x0 = constants.X / 2
    y0 = constants.Y / 2
    z0 = constants.d  # 物体点までの距離

    return np.array([[x0, y0, z0]], dtype=float)


def create_small_opening(
    constants: ClassicalConstants | QuantumConstants, width: int = 10
) -> np.ndarray:
    """
    画面中心に正方形状の小さな開口点群を生成する。

    Args:
        constants: 点群生成に使用する定数。
        width: 一辺あたりの点数。

    Returns:
        形状 `(width * width, 3)` の点群配列。
    """

    x0 = constants.X / 2
    y0 = constants.Y / 2
    z0 = constants.d  # 物体点までの距離

    offsets = np.arange(width) - (width - 1) / 2
    xs = x0 + offsets
    ys = y0 + offsets
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full_like(xx, z0)

    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return points


def create_four_points(constants: ClassicalConstants | QuantumConstants) -> np.ndarray:
    """
    画面中心の周囲に 4 点を配置した点群を生成する。

    Args:
        constants: 点群生成に使用する定数。

    Returns:
        形状 `(4, 3)` の点群配列。
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


def create_rectangle_points(
    constants: ClassicalConstants | QuantumConstants,
) -> np.ndarray:
    """
    四角形の輪郭に対応する点群を生成する。

    Args:
        constants: 点群生成に使用する定数。

    Returns:
        四角形輪郭上の点群配列。
    """

    x_size = constants.X // 2  # Xの1/2サイズの四角形を作る
    dx = np.array([constants.X / 4, 0.0, 0.0])
    dy = np.array([0.0, constants.Y / 4, 0.0])

    x_line = np.array([[x, constants.Y // 2, constants.d] for x in range(x_size)])
    y_line = np.array([[constants.X // 2, y, constants.d] for y in range(x_size)])

    # lineをスライドさせる TODO - numpy関数を使う
    top = x_line + dy + dx
    bottom = x_line - dy + dx
    left = y_line + dy + dx
    right = y_line + dy - dx

    rectangle = np.concatenate((top, bottom, left, right))
    return rectangle


def generate_hologram(points: np.ndarray, constants: ClassicalConstants) -> np.ndarray:
    """
    古典的な点群法でホログラムを計算する。

    Args:
        points: 物体点群。各要素は `(x, y, z)`。
        constants: ホログラム計算に使用する定数。

    Returns:
        形状 `(Y, X)` のホログラム配列。
    """
    x = np.arange(constants.X, dtype=np.float64) * constants.pp
    y = np.arange(constants.Y, dtype=np.float64) * constants.pp
    xx, yy = np.meshgrid(x, y)
    hologram = np.zeros((constants.Y, constants.X), dtype=np.float64)

    for xj, yj, zj in tqdm.tqdm(points):
        dx = xx - xj * constants.pp
        dy = yy - yj * constants.pp
        r = np.sqrt(dx * dx + dy * dy + zj * zj)
        hologram += np.cos(constants.k * r) / r
    return hologram


def show(holography: np.ndarray, X: int, Y: int) -> None:
    """
    ホログラム配列を等高線として表示する。

    Args:
        holography: 表示対象のホログラム配列。
        X: X 方向画素数。
        Y: Y 方向画素数。
    """
    fig, ax = plt.subplots()
    color = ax.contourf(range(X), range(Y), holography)
    fig.colorbar(color)
    fig.set_label("holography")
    plt.legend()
    plt.show()


def main():
    start = time.time()
    constants = ClassicalConstants()

    points = create_rectangle_points(constants)  # 四角形 # TODO - 分岐
    hologram = generate_hologram(points, constants)
    print("CGH Calculation completed!")

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))

    print("Preparing for display...")
    show(hologram, constants.X, constants.Y)


if __name__ == "__main__":
    main()

# TODO
# 1. コマンドライン引数を受け取れるようにする
# 2. Constantsに引数データを入れる
# 3. printの代わりにloggingを入れる
