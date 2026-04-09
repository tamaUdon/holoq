import time
from decimal import (
    Decimal,
    getcontext,
)
from fractions import Fraction

import numpy as np
import tqdm
from constants import ClassicalConstants
from pointcloud import create_single_point, show

### === Settings === ###
DEBUG = False
# numpy 固定小数モード
np.set_printoptions(precision=16, floatmode="fixed", suppress=False)
getcontext().prec = 16
np.set_printoptions(threshold=np.inf)  # type: ignore # ログを全て出す
### ================ ###


def _print_probabilities_unique_value(array: np.ndarray, name: str):
    """
    Numpy配列内の要素数をカウントし、出現確率をprintする
    """
    values, counts = np.unique(array, return_counts=True)
    probabilities = counts / array.size
    print(f"{name}の統計情報")
    for v, c, p in zip(values, counts, probabilities):
        print(f"要素: {v}, カウント: {c}, 確率: {p:.2f} \n")


def _extract_frac_part_from_theta(θ: np.ndarray) -> np.ndarray:
    """
    θ=M*N から小数部を取り出す
    - 整数部と小数部に分ける
    - 小数部を抜き出す
    - decimal型に変換

    return: 小数部<Decimal>
    """

    frac_part, int_part = np.modf(θ)  # 例) 1.5 -> (0.5, 1.0)
    decimal_arr = np.array(  # Decimal型に変換し、.以下をstrとして格納
        [[Decimal(str(x).split(".")[1]) for x in row] for row in frac_part],
        dtype=object,
    )
    return decimal_arr


def target(decimal_arr: np.ndarray) -> np.ndarray:
    """
    関数T(・)の実装
    - Decimal型の小数部を受け取る
    -
    """

    decimal_t_array = np.array(
        [[(Decimal(str(int(x))[0])) for x in row] for row in decimal_arr],
        dtype=int,
    )
    decimal_choice = np.where(
        decimal_t_array <= 4,
        0,  # 4以下を0に
        np.where(decimal_t_array >= 6, 1, 5),  # 6以上を1に, 5はそのまま
    )
    mask_5 = decimal_choice == 5  # 5の部分を特定
    count_5 = np.sum(mask_5)  # 5の数を数える
    decimal_choice[mask_5] = np.random.choice(
        [0, 1],  # 5 -> 0 or 1どちらかに振り分け
        size=count_5,
        p=[0.5, 0.5],
    )
    _print_probabilities_unique_value(decimal_t_array, "decimal_t_array")
    return decimal_choice


def monopolar_fixed_point(
    points: np.ndarray, constants: ClassicalConstants
) -> np.ndarray:
    """
    monopolar hologramの実装
    """
    x = np.arange(constants.X, dtype=np.int64)
    y = np.arange(constants.Y, dtype=np.int64)
    xh, yh = np.meshgrid(x, y)
    hologram = np.full((len(points), constants.X, constants.Y), 0)

    p_sq = 2 * np.pi * constants.pp * constants.pp
    p_denom = constants.λ * constants.d
    N = p_sq / p_denom  # noqa: N806
    print(f"{p_sq=}, {p_denom=}, {N=}")

    for xj, yj, _ in tqdm.tqdm(points):
        xhj = xh.astype(np.int32) - xj
        yhj = yh.astype(np.int32) - yj
        x_sq = xhj * xhj
        y_sq = yhj * yhj
        M = x_sq + y_sq  # M-bit # noqa: N806
        θ = M * N
        print(f"{M=} , {θ=}")

        decimal_arr = _extract_frac_part_from_theta(θ)
        hologram = target(decimal_arr)
    _print_probabilities_unique_value(hologram, "hologram")
    return hologram


def measure(N: int, hologram: np.ndarray) -> Fraction:  # noqa: N803
    """
    1をカウントし、ホログラム内の1の割合を返却する
    """
    count_one = np.count_nonzero(hologram == 1)
    ratio_of_one = Fraction(count_one.item(), N)
    print(f"{count_one=}, {ratio_of_one=}")

    return ratio_of_one


def random_hologram(
    ratio_of_one: Fraction,
    hologram: np.ndarray,
    constants: ClassicalConstants,
) -> np.ndarray:
    """
    measureの結果を元にホログラムのピクセルを1か0にフィルターする
    """
    rng = np.random.default_rng()
    random_filter = rng.random((constants.X, constants.Y))
    bool_filter = random_filter <= float(ratio_of_one)

    return bool_filter & hologram


def main():
    start = time.time()

    constants = ClassicalConstants()
    points = create_single_point(constants)
    hologram_raw = monopolar_fixed_point(points, constants)
    ratio_of_one = measure(N=constants.X * constants.Y, hologram=hologram_raw)

    hologram_rand = random_hologram(
        ratio_of_one=ratio_of_one, hologram=hologram_raw, constants=constants
    )

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))
    print("CGH Calculation completed!")

    print("Preparing for display...")
    show([hologram_raw, hologram_rand], constants.X, constants.Y)


if __name__ == "__main__":
    main()

# TODO - bin() を使って計算ができるようにする
# decimal型 -> 1~9 の整数が入ることはわかっている
#   -> 1~4...0, 6~9...1, 5 ... 0|1半々にする random()
