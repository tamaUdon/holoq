import math
import time
from datetime import datetime
from decimal import (
    getcontext,
)
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from constants import ClassicalConstants
from monoq_binary import extract_binary_frac_part_from_theta, target_binary
from monoq_decimal import (
    extract_decimal_frac_part_from_theta,
    target_decimal,
)
from pointcloud import (
    create_circle,
    create_four_points,
    create_rectangle_points,
    create_sin_wave,
    create_single_point,
    show,
)

### === Settings === ###
# デバッグモードフラグ
DEBUG = False
# 2進数モードフラグ
BINARY = True
# ターゲットビット
TARGET = 0
# 物体点
N = "4"  # "1", "4", "rectangle", "wave", "circle"
# 統計情報の保存先ディレクトリ
STATS_DIR = "results/stats/exp"
# 画像の保存先ディレクトリ
IMG_DIR = "results/images/monopolars/exp"
# numpy 固定小数モード
np.set_printoptions(precision=16, floatmode="fixed", suppress=False)
# ログを全て出力する設定
np.set_printoptions(threshold=np.inf)  # type: ignore
# Decimal 精度固定
getcontext().prec = 16
### ================ ###


### ====== Tools ====== ###
def _print_probabilities_unique_value(
    array: np.ndarray, name: str, dir: str = "", save: bool = False
):
    """
    Numpy配列内の要素数をカウントし、出現確率をprintする
    """

    values, counts = np.unique(array, return_counts=True)
    probabilities = counts / array.size
    # print(f"{name}の統計情報")
    # for v, c, p in zip(values, counts, probabilities):
    #    print(f"要素: {v}, カウント: {c}, 確率: {p:.2f} \n")

    if save:
        _save_probabilities(Path(dir), name, values, counts, probabilities)


def _save_probabilities(
    dir: Path,
    name: str,
    values: np.ndarray,
    counts: np.ndarray,
    probabilities: np.ndarray,
):
    """
    統計情報をファイルに保存する
    fname ... ファイルのパス
    """
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    fname = dir / (name + f"_{now}" + ".csv")

    if not dir.exists():
        dir.mkdir()

    with open(fname, "w", encoding="utf-8") as f:
        pd.DataFrame(
            {
                "value": values,
                "count": counts,
                "probability": probabilities,
            }
        ).to_csv(f, index=False)


def _monopolar_phase_from_point(
    xh: np.ndarray,
    yh: np.ndarray,
    xj: float,
    yj: float,
    zj: float,
    constants: ClassicalConstants,
) -> np.ndarray:
    dx = (xh - xj) * constants.pp
    dy = (yh - yj) * constants.pp
    return (math.pi / (constants.λ * zj)) * (dx * dx + dy * dy)


def monopolar_fixed_point(
    points: np.ndarray, constants: ClassicalConstants, binary: bool
) -> np.ndarray:
    """
    monopolar hologramの実装
    - points: 点群
    - constants: 定数オブジェクト
    - binary: 2進数版を実行するかどうかのフラグ
    """

    if binary:
        __extract_frac_part_from_theta = extract_binary_frac_part_from_theta
        __target = target_binary
    else:
        __extract_frac_part_from_theta = extract_decimal_frac_part_from_theta
        __target = target_decimal

    ### 固定小数実装版
    x = np.arange(constants.X, dtype=np.int32)
    y = np.arange(constants.Y, dtype=np.int32)
    xh, yh = np.meshgrid(x, y)
    holograms = []

    p_sq = np.pi * constants.pp * constants.pp
    p_denom = constants.λ
    N = p_sq / p_denom  # noqa: N806

    for idx, (xj, yj, zj) in enumerate(tqdm.tqdm(points)):
        xhj = xh.astype(np.int32) - xj
        yhj = yh.astype(np.int32) - yj
        x_sq = xhj * xhj
        y_sq = yhj * yhj
        z_sq = zj * zj

        M = x_sq + y_sq + z_sq  # M-bit # noqa: N806
        θ = M * N

        theta_frac = __extract_frac_part_from_theta(θ)
        holograms.append(__target(theta_frac, target=TARGET))

    return np.array(holograms)


def sum_holograms(holograms: np.ndarray, n: int) -> np.ndarray:
    holo_sum = holograms.sum(axis=0)
    holo_ratio = holo_sum / n  # 物体点数
    return holo_ratio


def random_hologram(
    holo_ratio: np.ndarray,
    constants: ClassicalConstants,
) -> np.ndarray:
    """
    measureの結果を元にホログラムのピクセルを1か0にフィルターする
    """

    rng = np.random.default_rng()
    random_filter = rng.random((constants.X, constants.Y))
    bool_filter = random_filter <= holo_ratio

    # _print_probabilities_unique_value(
    #     bool_filter,
    #     name=f"hologram_rand{TARGET}",
    #     dir=STATS_DIR,
    #     save=True,
    # )

    return bool_filter


### ====== Handler ====== ###
def main():
    start = time.time()

    constants = ClassicalConstants()

    if N == "1":
        points = create_single_point(constants)
    elif N == "4":
        points = create_four_points(constants)
    elif N == "rectangle":
        points = create_rectangle_points(constants)
    elif N == "wave":
        points = create_sin_wave(constants)
    elif N == "circle":
        points = create_circle(constants)
    else:
        raise NotImplementedError

    holograms = monopolar_fixed_point(points, constants, BINARY)
    holo_ratio = sum_holograms(holograms, len(points))  # /物体点
    hologram_rand = random_hologram(
        holo_ratio=holo_ratio, constants=constants
    )

    end = time.time()
    print(f"【{DEBUG=} 】")
    print(print("Cal time:{} sec".format(end - start)))
    print("CGH Calculation completed!")

    print("Preparing for display...")
    show(
        hologram_rand,
        constants.X,
        constants.Y,
        BINARY,
        TARGET,
        save=True,
        dir=Path(IMG_DIR),
    )


if __name__ == "__main__":
    main()


# TODO - z方向をX,Yと同じにする
# TODO - zを定数dを使わず、実際の奥行き値zを用いて実装する
# TODO - cibeやballなどの単純な点群データを用いて再構成まで検証する
# TODO - 奥行き方向に密（x,yが同じところに奥行き違いの点がある）な点群を作成する
# TODO - 奥行き方向に密な点群の、1000点の場合の画像を出す, マシンパワーが不足する場合はBrains
# TODO - bunnyなど3次元点群を入力する
# TODO - FFT-1でホログラムを再構成する (実際に動くか確認する)
# TODO - binaryとdecimalの処理を別ファイルにする
