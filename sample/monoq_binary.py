# 2進monopolarのツール

import numpy as np


def extract_binary_frac_part_from_theta(θ: np.ndarray) -> np.ndarray:
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
    ).reshape(*frac_part.shape, 8)  # 例) [3] -> [1,1]

    return binary_frac


def target_binary(theta_frac: np.ndarray, target: int | None) -> np.ndarray:
    """
    関数T(・)の実装
    - binaryにしたθの小数部を受け取る (big endian)
    - 任意の桁を取り出し、0 or 1の配列をつくって返却する
    """
    # [:,0]...1文字目を取り出す (big endian最上位の桁)
    target_bit = int(f"{target}")
    binary_choice = theta_frac[
        :, :, target_bit
    ]  # 全ての行の各列1文字目を抽出
    # _print_probabilities_unique_value(
    #     theta_frac[:, :, target_bit], name=f"theta_frac[:, :, {target_bit}]"
    # )
    return binary_choice
