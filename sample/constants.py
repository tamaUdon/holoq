from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ClassicalConstants:
    """
    古典CGH計算で使用する定数群。

    Attributes:
        X: ホログラムの X 方向画素数。
        Y: ホログラムの Y 方向画素数。
        λ: 波長 [m]。
        pp: 画素ピッチ [m]。
        d: ホログラム面と物体面の距離 [m]。
    """

    X = 256  # 画素X方向
    Y = X
    λ = 633e-9  # 波長[nm] e-9
    pp = 10e-6  # 画素ピッチ[μm] e-6
    d = 80e-3  # 物体までの距離[mm] e-3 # 260 # TODO - 自動計算に変更
    bits_w = 2

    @property
    def k(self) -> float:
        return 2 * math.pi / self.λ

    @property
    def pad(self) -> int:
        # X幅のpaddingを追加
        return self.X


@dataclass
class QuantumConstants:
    """
    量子CGH回路で使用する定数群。

    Attributes:
        N: 物体点数。
        X: ホログラムの X 方向画素数。
        W: 量子回路で使用する基準ビット幅。
        Y: ホログラムの Y 方向画素数。`__post_init__` で `X` に合わせる。
        obj_w: 物体点インデックスの表現ビット幅。
        xh_w: ホログラム面 x 座標の表現ビット幅。
        yh_w: ホログラム面 y 座標の表現ビット幅。
        diff_x_w: x 方向差分計算に必要なビット幅。
        diff_y_w: y 方向差分計算に必要なビット幅。
        SHAPE: 物体形状の識別子。
        TEST: テスト用フラグ。
    """

    N: int  # 4
    X: int  # 15
    W: int = 0  # ビット幅
    Y: int = 0  # Y = X

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
        self.obj_w = math.ceil(math.log2(self.N))
        self.xh_w = math.ceil(math.log2(self.X))
        self.yh_w = math.ceil(math.log2(self.Y))
        self.W = max(self.obj_w, self.xh_w, self.yh_w)
