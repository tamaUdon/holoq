import dataclasses
from dataclasses import dataclass
import math
import numpy as np


@dataclasses.dataclass(frozen=True)
class ClassicalConstants:
    """
    古典Constants の Docstring

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
    """

    X = 512  # 画素X方向
    Y = X
    λ = 633e-9  # 波長[nm]
    pp = 10e-6  # 画素ピッチ[μm]
    d = 260e-3  # 物体までの距離[mm]

    @property
    def k(self) -> float:
        return 2 * math.pi / self.λ

    @property
    def pad(self) -> int:
        # X / 2 幅のpadding
        return self.X // 2


@dataclasses.dataclass(frozen=True)
class QuantumConstants:
    """
    量子Constants の Docstring

    :param N: 物体点数 ダミーを含む 2の累乗
    :type N: int
    :param a: 振幅 ※初期的な実装では不要?
    :type a: int (nm)
    :param ρ: 位相のリスト
    :type ρ: np.ndarray[float]
    :param xj_yj: 物体点の座標 (x,y)
    :type xj_yj: np.ndarray[int]
    :param xh_yh: ホログラムの座標 (x,y)
    :type xh_yh: np.ndarray[int]
    """

    N = 4
    X = 100
    Y = X
    TEST: bool = True
    shape: str = "point"  # "circle" | "square" | "point"

    # 残りのTODO
    # 2. 4点 (四角), 円 (多数点)から作成できるか試したい
    # 3. ゾーンプレートが確かめられたら


@dataclass
class BitWidth:
    b_width: int  # 2

    @property
    def add_w(self) -> int:
        return self.b_width + 1  # 3

    @property
    def mul_w(self) -> int:
        return self.add_w + self.add_w  # 6

    @property
    def sq_w(self) -> int:
        return self.mul_w + 1  # 7

    @property
    def res_w(self) -> int:
        return self.sq_w + 1  # 8
