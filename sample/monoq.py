import numpy as np
import tqdm
import time
from constants import ClassicalConstants
from pointcloud import create_single_point, show
from decimal import Decimal, getcontext


# numpy を固定小数モードに
np.set_printoptions(precision=16, floatmode='fixed', suppress=False)
getcontext().prec = 16

def monopolar_fixed_point(points: np.ndarray, constants: ClassicalConstants):
    # Complex, amplitude and phase-only holograms using bipolar approximationのFig.2を参考に作成

    x = np.arange(constants.X, dtype=np.int64)
    y = np.arange(constants.Y, dtype=np.int64)
    xh, yh = np.meshgrid(x, y)
    hologram = np.full((len(points), constants.X, constants.Y), 0) # 0埋めのhologram面 * 物体点数

    p_sq = 2 * np.pi * constants.pp * constants.pp
    p_denom = constants.λ * constants.d

    # 1. 固定小数monopolarを実装

    N = p_sq / p_denom # M-bit 固定値なのでループの外側に出す
    print(f"{p_sq=}")
    print(f"{p_denom=}")
    print(f"{N=}") # N=0.0003038036213391664
    #  TODO - ... order of nanometers, 𝑝2∕2𝜆𝑧𝑗 is a fractional value. If this is converted to 𝑀-bit fixed point format, the fixed-point is always positioned on the leftmost side

    for i, (xj, yj, zj) in enumerate(tqdm.tqdm(points)):
        xhj = xh.astype(np.int32) - xj # hologram面を一気に計算
        yhj = yh.astype(np.int32) - yj
        print(f"{xj=}, {yj=}, {zj=}")
        print(f"{xhj=}")
        print(f"{yhj=}")

        x_sq = xhj * xhj
        y_sq = yhj * yhj
        M = x_sq + y_sq # M-bit

        print(f"{x_sq=}")
        print(f"{y_sq=}")
        print(f"{M=}")

        θ = M * N # θ
        print(f"{θ=}")

        # 1. fixed-point monopolar generated holography
        frac_part, int_part = np.modf(θ) # 例) 1.5 -> (0.5, 1.0)
        print(f"{frac_part=}")
        print(f"{int_part=}")

        decimal_arr = np.array([[Decimal(str(x).split(".")[1]) for x in row] for row in frac_part], dtype=object) # Decimal型に変換し、.以下をstrとして格納
        
        t_array = np.array([[(Decimal(str(int(3))[0])/10).quantize(Decimal('0'), ROUND_HALF_UP) for x in row] for row in decimal_arr], dtype=int) # 不格好なのでなおす
        # decimal型 -> 0~9 整数が入ることはわかっている　-> 0~
        # 四捨五入はok
        print(f"{t_array=}")
 
        hologram = t_array  # t
        print(f"{hologram=}")
        # hologram[i] = t.astype(int) # 足し合わせている -> 個別に保持する
        # print(hologram[i])

        # TODO - 2. calculate the ratio of 1
        #  - T(・) ... 小数第一位を取得する関数を作り、1の数を数える
        #    例) 75%などの確率を得る

        # TODO - (3. calculate the ratio after measurement)
        
        # TODO - 3. hologram[] * measurement ratio 

    return hologram


def main():

    start = time.time()

    constants = ClassicalConstants()
    points = create_single_point(constants)
    hologram = monopolar_fixed_point(points, constants)

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))
    print("CGH Calculation completed!")

    print("Preparing for display...")
    show(hologram, constants.X, constants.Y)


if __name__ == "__main__":
    main()
    


# <処理フローの詳細>
# 1. ホログラムの計算部 -> oks
#     - 古典CGHと同じ、Monopolar hologram computation を用いる

# 2. 各ピクセルの輝度値の計算部
#     - T(・) ... 小数第一位を取得する関数を作り、1の数を数える
#     - 点の物体点の場合、等確率（重み1/√4）の量子重ね合わせになっていますが、
#       T(・)で計測した結果、|1＞が3つの場合は　計算結果1が　3/4　の確率で得られます。輝度値は75%になるわけではありません。
#       多qubitから特定のビットのみを測定した場合の確率計算はお貸ししている以下の書籍の8.5節を参照してください。

# 3. ノイズ計算
#     - random() などで確率的に1が得られる実装とする
#     - 例) 各ピクセルの正しい輝度値が75%の確率で得られる

# 4. 結果の足し合わせ部分
#     - 確率的に計算した輝度値を足し合わせる
#     - 全ピクセル分計算すると、確率的なホログラム像が得られる

# 5. 上記のコードを用いた検証部分
#     - コードが完成次第、ご相談させてください。

# source .venv/bin/activate