import numpy as np
import tqdm
import time
from constants import ClassicalConstants
from pointcloud import create_single_point, show


def monopolar(points: np.ndarray, constants: ClassicalConstants):
    # Complex, amplitude and phase-only holograms using bipolar approximationのFig.2を参考に作成
    # monopolar実装版 - 512*512画素で3sec
    scale = 1 << ClassicalConstants.bits_w
    target_bit = ClassicalConstants.bits_w - 1

    x = np.arange(ClassicalConstants.X, dtype=np.int64)
    y = np.arange(ClassicalConstants.Y, dtype=np.int64)
    xh, yh = np.meshgrid(x, y)
    hologram = np.full((len(points), constants.X, constants.Y), 0) # 0埋めのhologram面 * 物体点数

    p_sq = ClassicalConstants.pp * ClassicalConstants.pp
    p_denom = 2 * ClassicalConstants.λ * ClassicalConstants.d
    M = p_sq / p_denom # M-bit 固定値なのでループの外側に出す

    for i, (xj, yj, zj) in enumerate(tqdm.tqdm(points)):
        xhj = xh.astype(np.float64) - xj # hologram面を一気に計算
        yhj = yh.astype(np.float64) - yj
        print(f"{xhj=}")
        print(f"{yhj=}")

        x_sq = xhj * xhj
        y_sq = yhj * yhj
        N = x_sq + y_sq # N-bit
        print(f"{x_sq=}")
        print(f"{y_sq=}")
        print(f"{N=}")

        # fixed-point
        # TODO - numpy 配列全体に桁数を知る計算をする
        num_of_digits = len(str(abs(N))) # 桁数を知る. eg. 3桁
        print(f"{num_of_digits=}")

        N_decimal = N / 10**num_of_digits # 111 / 10**3 = 0.111
        print(f"{N_decimal=}")

        ρ = M * N_decimal
        print(f"{ρ=}")
        t = (ρ >> target_bit) & 1 # numpy 配列全体にTをかける
        print(f"{t=}")

        hologram[i] += t.astype(np.float64) # 足し合わせている -> 個別に保持する
        print(hologram[i])
    return hologram


def main():
    start = time.time()

    constants = ClassicalConstants()
    points = create_single_point(constants)
    hologram = monopolar(points, constants)

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))
    print("CGH Calculation completed!")

    print("Preparing for display...")
    show(hologram)


if __name__ == "__main__":
    main()
    

# <処理フローの詳細>
# 1. ホログラムの計算部
#     - 古典CGHと同じ、Monopolar hologram computation を用いる

# 2. 各ピクセルの輝度値の計算部
#     - T(・) ... 小数第一位を取得する関数を作り、1の数を数える
#     - 例) 3/4のビットが1の時 ... 輝度値は75%

# 3. ノイズ計算
#     - random() などで確率的に1が得られる実装とする
#     - 例) 各ピクセルの正しい輝度値が75%の確率で得られる

# 4. 結果の足し合わせ部分
#     - 確率的に計算した輝度値を足し合わせる
#     - 全ピクセル分計算すると、確率的なホログラム像が得られる

# 5. 上記のコードを用いた検証部分
#     - コードが完成次第、ご相談させてください。