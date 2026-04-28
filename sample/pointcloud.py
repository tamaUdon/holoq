# 点群法でホログラムを表示するスクリプト
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from constants import ClassicalConstants
import pandas as pd
from pathlib import Path


def create_single_point(constants: ClassicalConstants) -> np.ndarray:
    """
    create_single_point 1点の点群を作成する関数

    - X*Yの中心に物体点 (1点) がある想定

    :param constants: 定数クラスのオブジェクト
    :type constants: ClassicalConstants
    :return: デバッグ用の物体点 (1点)
    :rtype: np.ndarray
    """
    x0 = constants.X / 2
    y0 = constants.Y / 2
    z0 = constants.d  # 物体点までの距離

    return np.array([[x0, y0, z0]], dtype=float)


def create_small_opening(
    constants: ClassicalConstants, width: int = 10
) -> np.ndarray:
    """
    create_small_opening 小さな開口部の点群を作成する関数

    - X*Yの中心に物体点 (10*10点) がある想定

    :param constants: 定数クラスのオブジェクト
    :type constants: ClassicalConstants
    :param width: int
    :typewidth: 開口部の幅
    :return: デバッグ用の物体点 (1点)
    :rtype: np.ndarray
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


def create_circle(constants: ClassicalConstants) -> np.ndarray:
    """
    create_circle 円形の点群を作成する関数

    :param constants: 定数クラスのオブジェクト
    :type constants: ClassicalConstants
    :return: 円形の点群
    :rtype: np.ndarray
    """

    scale = 6.0
    x = np.arange(constants.X, dtype=np.float64)
    y = np.arange(constants.Y, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    cx = (constants.X - 1) / scale
    cy = (constants.Y - 1) / scale
    radius_sq = (constants.X * constants.Y) / (scale * np.pi)

    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_sq
    points_xy = np.column_stack((xx[mask], yy[mask]))
    z = np.full((points_xy.shape[0], 1), constants.d, dtype=np.float64)

    return np.hstack((points_xy, z))


def create_four_points(constants: ClassicalConstants) -> np.ndarray:
    """
    create_rect_points 4点の点群を作成する関数

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


def create_rectangle_points(constants: ClassicalConstants) -> np.ndarray:
    """
    create_rectangle_points 四角形の点群を作成する関数

    :param constants: 定数クラスのオブジェクト
    :type constants: Constants
    """

    x_size = constants.X // 2  # Xの1/2サイズの四角形を作る
    dx = np.array([constants.X / 4, 0.0, 0.0])
    dy = np.array([0.0, constants.Y / 4, 0.0])

    x_line = np.array(
        [[x, constants.Y // 2, constants.d] for x in range(x_size)]
    )
    y_line = np.array(
        [[constants.X // 2, y, constants.d] for y in range(x_size)]
    )

    # lineをスライドさせる TODO - numpy関数を使う
    top = x_line + dy + dx
    bottom = x_line - dy + dx
    left = y_line + dy + dx
    right = y_line + dy - dx

    rectangle = np.concatenate((top, bottom, left, right))
    return rectangle


def create_depth_line(
    depth_n: int,
    constants: ClassicalConstants,
    depth: float = 30e-3,
) -> np.ndarray:
    # 奥行方向のみにdepth_n個の点群を打つ
    if depth_n <= 0:
        raise ValueError("depth_n must be greater than 0")
    if depth < 0:
        raise ValueError("depth must be greater than or equal to 0")

    x = np.full(depth_n, constants.X / 2, dtype=np.float64)
    y = np.full(depth_n, constants.Y / 2, dtype=np.float64)

    if depth_n == 1 or depth == 0:
        z = np.full(depth_n, constants.d, dtype=np.float64)
    else:
        z = np.linspace(
            constants.d - depth / 2.0,
            constants.d + depth / 2.0,
            depth_n,
            dtype=np.float64,
        )

    return np.column_stack((x, y, z))


def create_cube(
    constants: ClassicalConstants,
    width: int | None = None,
    depth: float = 30e-3,
    step: int = 8,
) -> np.ndarray:
    """
    create_cube cube型の点群を作成する関数

    - x, y はホログラム面の画素座標。
    - z はホログラム面から物体点までの距離 [m]。
    - 表面だけでなく、中身にも点が詰まった立方体を作る。

    :param constants: 定数クラスのオブジェクト
    :type constants: ClassicalConstants
    :param width: cubeのx/y方向サイズ [pixel]
    :type width: int | None
    :param depth: cubeのz方向サイズ [m]
    :type depth: float
    :param step: x/y/z方向の点間隔
    :type step: int
    :return: cube型の点群
    :rtype: np.ndarray
    """

    if step <= 0:
        raise ValueError("step must be greater than 0")

    if width is None:
        width = constants.X // 4
    if width <= 0:
        raise ValueError("width must be greater than 0")

    cx = (constants.X - 1) / 2.0
    cy = (constants.Y - 1) / 2.0
    half_width = width / 2.0

    xs = np.arange(cx - half_width, cx + half_width + 1, step)
    ys = np.arange(cy - half_width, cy + half_width + 1, step)

    z_step = step * constants.pp
    if depth == 0:
        zs = np.array([constants.d], dtype=np.float64)
    else:
        z_count = max(2, int(np.floor(depth / z_step)) + 1)
        zs = np.linspace(
            constants.d - depth / 2.0,
            constants.d + depth / 2.0,
            z_count,
            dtype=np.float64,
        )

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))


def create_sin_wave(constants: ClassicalConstants, debug=False) -> np.ndarray:
    """
    create_sin_wave sin波の点群を作成する関数

    :param constants: 定数クラスのオブジェクト
    :type constants: Constants
    """
    x = np.arange(constants.X, dtype=np.float64)
    center_y = constants.Y / 2
    y = center_y + (constants.Y / 10) * np.sin((6 * np.pi / constants.X) * x)
    z = np.full(
        constants.X, constants.d, dtype=np.float64
    )  # zは全てconstants.d

    if debug:
        fig, ax = plt.subplots()
        ax.plot(x, y, linewidth=1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Sin Wave Pointcloud")
        plt.tight_layout()
        plt.show()

    return np.column_stack((x, y, z))


def generate_hologram(
    points: np.ndarray, constants: ClassicalConstants
) -> np.ndarray:
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


def show(
    holography: np.ndarray | list[np.ndarray],
    x: int,
    y: int,
    binary: bool,
    target: int = 0,
    dir: Path = Path(),
    save: bool = False,
) -> None:
    """
    ホログラム配列を等高線として表示する。
    1枚または複数枚のホログラムを並べて表示する。

    Args:
        holography: 表示対象のホログラム配列、または配列のリスト
        X: X 方向画素数
        Y: Y 方向画素数
    """

    if isinstance(holography, np.ndarray):
        holography_list = [holography]
    else:
        holography_list = holography

    n = len(holography_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    if n == 1:
        axes = [axes]

    label = "hologram_Decimal"
    if binary:
        label = "hologram_Binary"

    for i, holo in enumerate(holography_list):
        color = axes[i].contourf(range(x), range(y), holo)
        cbar = fig.colorbar(color, ax=axes[i])
        cbar.set_label(label)
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")

    if save:
        if not dir.exists():
            dir.mkdir()
        now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        outpath = dir / (f"monopolar_{label}_t{target}" + f"_{now}" + ".png")
        plt.savefig(outpath, dpi=350, bbox_inches="tight")

    plt.tight_layout()
    plt.show()


def main():
    binary = False  # 10進
    start = time.time()
    constants = ClassicalConstants()

    points = create_sin_wave(constants, debug=False)  # 四角形 # TODO - 分岐
    hologram = generate_hologram(points, constants)
    print("CGH Calculation completed!")

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))

    print("Preparing for display...")
    show(hologram, constants.X, constants.Y, binary)


if __name__ == "__main__":
    main()

# TODO
# 1. コマンドライン引数を受け取れるようにする
# 2. Constantsに引数データを入れる
# 3. printの代わりにloggingを入れる
