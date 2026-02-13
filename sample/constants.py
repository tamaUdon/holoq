<<<<<<< HEAD
import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class Constants:
    """
    Constants の Docstring

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

    :return: bipolarホログラムの計算結果
    :rtype: np.ndarray
    """

    X = 512  # 画素X方向
    Y = X
    λ = 633e-9  # 波長[nm]
    pp = 10e-6  # 画素ピッチ[μm]
    d = 260e-3  # 物体までの距離[mm]
    bits_w = 16

    @property
    def k(self) -> float:
        return 2 * math.pi / self.λ

    @property
    def pad(self) -> int:
        # X / 2 幅のpadding
        return self.X // 2
=======
import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class Constants:
    """
    Constants の Docstring

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

    :return: bipolarホログラムの計算結果
    :rtype: np.ndarray
    """

    X = 512  # 画素X方向
    Y = X
    λ = 633e-9  # 波長[nm]
    pp = 10e-6  # 画素ピッチ[μm]
    d = 260e-3  # 物体までの距離[mm]
    bits_w = 4

    @property
    def k(self) -> float:
        return 2 * math.pi / self.λ

    @property
    def pad(self) -> int:
        # X / 2 幅のpadding
        return self.X // 2
>>>>>>> beae84c94878fe173af18f90a654f1f6540ed53b
