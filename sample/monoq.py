import time
from datetime import datetime
from decimal import (
    Decimal,
    getcontext,
)
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from constants import ClassicalConstants
from pointcloud import (
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
    print(f"{name}の統計情報")
    for v, c, p in zip(values, counts, probabilities):
        print(f"要素: {v}, カウント: {c}, 確率: {p:.2f} \n")

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


def _extract_decimal_frac_part_from_theta(θ: np.ndarray) -> np.ndarray:
    """
    10進数ver. θ=M*N から小数部を取り出す
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


def _extract_binary_frac_part_from_theta(θ: np.ndarray) -> np.ndarray:
    """
    2進数ver. θ=M*N から小数部を取り出す
    - 整数部と小数部に分ける
    - 小数部を抜き出す
    - binaryに変換

    return: 小数部<Decimal>
    """

    frac_part, int_part = np.modf(θ)  # 例) 1.5 -> (0.5, 1.0)
    frac_scaled = (frac_part * 255).astype(
        np.uint8  # uint8 に変換
    )  # unpackbits は uint8 のみ対応

    binary_frac = np.unpackbits(  # 2進数に変換
        frac_scaled, axis=1, bitorder="big"
    ).reshape(512, 512, 8)  # 例) [3] -> [1,1]

    return binary_frac


def _target_decimal(decimal_arr: np.ndarray, idx: int) -> np.ndarray:
    """
    関数T(・)の実装
    - Decimal型の小数部を受け取る
    - 0 or 1半々に振り分けて返却する
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

    _print_probabilities_unique_value(
        decimal_t_array,
        name=f"decimal_t_array_p{idx}",  # TODO - 引数かconstantsから受け取る
        dir=STATS_DIR,
        save=True,
    )
    return decimal_choice


def _target_binary(theta_frac: np.ndarray, idx: int) -> np.ndarray:
    """
    関数T(・)の実装
    - binaryにしたθの小数部を受け取る (big endian)
    - 任意の桁を取り出し、0 or 1の配列をつくって返却する
    """
    print(theta_frac.shape)

    # [:,0]...1文字目を取り出す (big endian最上位の桁)
    binary_choice = theta_frac[:, :, TARGET]  # 全ての行の各列1文字目を抽出
    _print_probabilities_unique_value(
        theta_frac[:, :, TARGET], name=f"theta_frac[:, :, {TARGET}]"
    )

    _print_probabilities_unique_value(
        binary_choice,
        name=f"binary_choice_t{TARGET}_p{idx}",
        dir=STATS_DIR,
        save=True,
    )
    return binary_choice


def monopolar_fixed_point(
    points: np.ndarray, constants: ClassicalConstants, binary: bool
) -> np.ndarray:
    """
    monopolar hologramの実装 10進数
    - points: 点群
    - constants: 定数オブジェクト
    - binary: 2進数版を実行するかどうかのフラグ
    """

    if binary:
        __extract_frac_part_from_theta = _extract_binary_frac_part_from_theta
        __target = _target_binary
    else:
        __extract_frac_part_from_theta = _extract_decimal_frac_part_from_theta
        __target = _target_decimal

    if DEBUG:
        """ 
        ## float実装版, 比較用  monopolar.monopolar_numpy() より
        # WARNING - うまく表示されてない (6*6のzoneplate) 
        #           np.cos()を用いたmonopolarの場合は問題ない
        # TODO - pp, d, λのかけ方を確認する
        """
        x = np.arange(constants.X, dtype=np.float64) * constants.pp
        y = np.arange(constants.Y, dtype=np.float64) * constants.pp
        xx, yy = np.meshgrid(x, y)
        holograms = np.zeros((constants.Y, constants.X), dtype=np.float64)

        for idx, (xj, yj, zj) in enumerate(tqdm.tqdm(points)):
            hx = xx - xj * constants.pp
            hy = yy - yj * constants.pp
            rho = constants.k / zj
            phase = rho * (hx * hx + hy * hy + zj * zj)

            theta_frac = __extract_frac_part_from_theta(phase)
            holograms += __target(theta_frac, idx)

        print(f"{holograms=}")

    else:
        ### 固定小数実装版
        x = np.arange(constants.X, dtype=np.int32)
        y = np.arange(constants.Y, dtype=np.int32)
        xh, yh = np.meshgrid(x, y)
        holograms = []
        ratio_after_measure = Fraction()

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
            t = __target(theta_frac, idx=idx)
            ratio_of_one, ratio_after_measure = measure(
                N=len(points), hologram=t
            )
            hologram_rand = random_hologram(
                ratio=ratio_of_one,
                hologram=t,
                constants=constants,
                idx=idx,
            )
            holograms.append(hologram_rand)

    return holograms


def measure(N: int, hologram: np.ndarray) -> tuple[Fraction, float]:  # noqa: N803
    """
    1をカウントし、ホログラム内の1の割合を返却する
    """
    count_one = np.count_nonzero(hologram)
    ratio_of_one = Fraction(
        count_one.item(), N
    )  # Q - 物体点が1の時、/1となる？ 重ね合わせ状態の定義を確認する?
    sqrt_ratio_of_one = np.sqrt(count_one.item()) / np.sqrt(N)
    ratio_after_measure = (
        1 / np.sqrt(N)
    ) / sqrt_ratio_of_one  # TODO - 正しい式と結果かを確認する

    # TODO - 物体点が複数個ある場合、測定後の確率計算を入れる

    print(f"{count_one=}, {ratio_of_one=}, {ratio_after_measure=}")

    return ratio_of_one, ratio_after_measure


def random_hologram(
    ratio: Fraction,
    hologram: np.ndarray,
    constants: ClassicalConstants,
    idx: int,
) -> np.ndarray:
    """
    measureの結果を元にホログラムのピクセルを1か0にフィルターする
    """

    rng = np.random.default_rng()
    random_filter = rng.random((constants.X, constants.Y))
    bool_filter = random_filter <= float(ratio)
    hologram_rand = bool_filter & hologram.astype(np.int64)

    print(f"{ratio=}")
    # print(f"{bool_filter=}")

    _print_probabilities_unique_value(
        hologram_rand,
        name=f"hologram_rand{TARGET}_p{idx}",
        dir=STATS_DIR,
        save=True,
    )

    return hologram_rand


### ====== Handler ====== ###
def main():
    start = time.time()

    constants = ClassicalConstants()
    points = create_single_point(constants)
    # points = create_four_points(constants)
    # points = create_rectangle_points(constants)
    holograms = monopolar_fixed_point(points, constants, BINARY)

    end = time.time()
    print(f"【{DEBUG=} 】")
    print(print("Cal time:{} sec".format(end - start)))
    print("CGH Calculation completed!")

    print("Preparing for display...")
    show(
        holograms,
        constants.X,
        constants.Y,
        BINARY,
        TARGET,
        save=True,
        dir=Path(IMG_DIR),
    )


if __name__ == "__main__":
    main()

# TODO - 物体点数を1,4,四角と増やす -> ok
# TODO - ホログラムの出力が正しいか？ゾーンプレートらしい点が多すぎる -> ok
#   1点の時5*5, 4点と四角の時6*6 -> 修正済み -> ok
# TODO - 再度  物体点数を1,4,四角と増やす -> ok. rectangleは表示点数が多すぎてshow()できなかった
# TODO - count_one, ratio_of_one がおかしいので修正する -> ok
# TODO - bunnyなど3次元点群を入力する
#   pointcloud.py
#   monoq.py 両方で確かめる
# TODO - 1にする部分は1回だけにする -> ok,
# TODO - decimal と binary を比較する -> ok
# TODO - 1点と4点の場合を比較する
# TODO - 物体点が複数個ある場合、測定後の確率計算を入れる
# TODO - ランダムの画質を確認する
