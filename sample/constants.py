import dataclasses
import math


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

    :param N: 物体点数 ダミーを含む
    :type N: int
    :param a: 振幅 ※初期的な実装では不要?
    :type a: int (nm)
    :param ρ: 位相のリスト
    :type ρ: list[float]
    :param xj_yj: 物体点の座標 (x,y)
    :type xj_yj: list[int]
    :param xh_yh: ホログラムの座標 (x,y)
    :type xh_yh: list[int]
    """

    N = 4
    bits_w = 2
    a = [1, 2, 3, 0]
    ρ = [0.5, 0.25, 0.5, 0]  # 最後ρ=0 ... 位相の寄与なし=ダミー
    xj_yj = [(0, 0), (1, 0), (0, 1), (1, 1)]
    xh_yh = [(0, 0), (1, 0), (0, 1), (1, 1)]
