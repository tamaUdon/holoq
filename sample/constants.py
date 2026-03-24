import dataclasses
from dataclasses import dataclass
import math
import numpy as np
from typing import Optional


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


@dataclasses.dataclass
class QuantumConstants:
    """
    量子Constants の Docstring

    :param N: 物体点数 ダミーを含む 2の累乗
    :type N: int
    :param X: 物体点の座標 (X軸)
    :type X: int
    :param Y: ホログラムの画素数 (Y軸)
    :type Y: int
    :param TEST: テストフラグ
    :type TEST: bool
    :param shape: 物体の形状 (2D)
    :type shape: str
    """

    N: int  # 4
    X: int  # 15
    W: int = 0  # W = N
    Y: int = 0  # Y = X

    obj_w: int = 0
    xh_w: int = 0
    yh_w: int = 0
    diff_x_w: int = 0
    diff_y_w: int = 0

    SHAPE: str = "point"  # "circle" | "square" | "point"
    TEST: bool = True

    @property
    def add_w(self) -> int:
        return self.W + 1  # 3

    @property
    def mul_w(self) -> int:
        return self.add_w + self.add_w  # 6

    @property
    def sq_w(self) -> int:
        return self.mul_w + 1  # 7

    @property
    def res_w(self) -> int:
        return self.sq_w + 1  # 8

    def __post_init__(self) -> None:
        self.Y = self.X
        self.W = self.N
        self.obj_w = math.ceil(math.log2(self.N))
        self.xh_w = math.ceil(math.log2(self.X))
        self.yh_w = math.ceil(math.log2(self.Y))
        self.diff_x_w = max(self.obj_w, self.xh_w) + 1
        self.diff_y_w = max(self.obj_w, self.yh_w) + 1

    # 残りのTODO
    # 2. 4点 (四角), 円 (多数点)から作成できるか試したい
    # 3. ゾーンプレートが確かめられたら
