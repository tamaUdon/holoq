# 10進monopolarのツール

from decimal import (
    Decimal,
)

import numpy as np
import pandas as pd


def extract_decimal_frac_part_from_theta(θ: np.ndarray) -> np.ndarray:
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


def target_decimal(decimal_arr: np.ndarray, target: int = -1) -> np.ndarray:
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

    # _print_probabilities_unique_value(
    #     decimal_t_array,
    #     name=f"decimal_t_array_p{idx}",  # TODO - 引数かconstantsから受け取る
    #     dir=STATS_DIR,
    #     save=True,
    # )
    return decimal_choice
