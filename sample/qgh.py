# 点群法で量子コンピュータ生成ホログラムを作る

import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from constants import Constants
from sample.pointcloud import create_single_point, generate_hologram, show


# 古典点群法
def monopolar_numpy(points: np.ndarray, constants: Constants):
    # numpy実装版 - 512*512画素で7sec
    x = np.arange(constants.X, dtype=np.float64) * constants.pp
    y = np.arange(constants.Y, dtype=np.float64) * constants.pp
    xx, yy = np.meshgrid(x, y)
    hologram = np.zeros((constants.Y, constants.X), dtype=np.float64)

    for xj, yj, zj in tqdm.tqdm(points):
        hx = xx - xj * constants.pp
        hy = yy - yj * constants.pp
        rho = constants.k / zj
        phase = rho * (hx * hx + hy * hy + zj * zj)
        hologram += np.where(np.cos(phase) >= 0.0, 1.0, -1.0)

    return hologram


# 量子ビットに埋め込み
def embed(points: np.ndarray, constants: Constants):
    # 量子レジスタを初期化
    a_qbit = ...
    P_qbit = ...
    x_qbit = ...
    y_qbit = ...

    for xj, yj, zj in tqdm.tqdm(points):
        # xj, yj (zj) にアダマールゲートをかけて、重ね合わせ状態にする
        hadamard()
        ...
        # a_j, rho_j, hx, hyにControlled-NOTをかけてqbitにし、重ね合わせ状態にする
        CNOT()
        ...

    if len(points) % 2 != 0:
        # 点群数が奇数ではないとき、2^2となる最小の数を求め、
        # 不足しているビット数ぶん0（ダミー）を入れる
        ...


# アダマール行列で重ね合わせ
def hadamard(): ...


# Controlled-NOT
def CNOT(): ...


# 量子回路を作る
ADD = ...
MUL = ...


# T(・)で測定
def T(): ...


# 1の個数を数える
def count(): ...


def main():
    start = time.time()
    constants = Constants()

    points = create_single_point(constants)  # 四角形 # TODO - 分岐
    hologram = generate_hologram(points, constants)
    print("CGH Calculation completed!")

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))

    print("Preparing for display...")
    show(hologram)


if __name__ == "__main__":
    main()
