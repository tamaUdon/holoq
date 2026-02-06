# 点群法で量子コンピュータ生成ホログラムを作る

import time
import math
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Iterable, Tuple
from constants import Constants
from pointcloud import (
    create_single_point,
    create_rectangle_points,
    generate_hologram,
    show,
)
from constants import Constants

# Qiskit (Aer)
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

bits_width = Constants.bits_w


def prepare_basis_state(n: int, k: int) -> QuantumCircuit:
    """
    基底状態を作る

    :param n: 量子ビットの数
    :type n: int
    :param k: 初期状態の値
    :type k: int
    :return: 量子回路
    :rtype: QuantumCircuit
    """
    qc = QuantumCircuit(n)
    for i in range(n):
        if (k >> i) & 1:
            qc.x(i)
    return qc


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
def encode_basis(points: np.ndarray, constants: Constants):
    # 量子レジスタを初期化
    a_qbit = ...
    P_qbit = ...
    x_qbit = ...
    y_qbit = ...

    for xj, yj, zj in tqdm.tqdm(points):
        # xj, yj (zj) にアダマールゲートをかけて、重ね合わせ状態にする
        Hadamrd_matrix()
        ...
        # a_j, rho_j, hx, hyにControlled-NOTをかけてqbitにし、重ね合わせ状態にする
        CNOT()
        ...

    if len(points) % 2 != 0:
        # 点群数が奇数ではないとき、2^2となる最小の数を求め、
        # 不足しているビット数ぶん0（ダミー）を入れる
        ...


# アダマール行列
# 重ね合わせをつくる行列
Hadamrd_matrix = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

# Controlled-Notゲート
# 標的ビットを制御するゲート
CNOT_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


# Controlled-NOT
def CNOT(): ...


# Hadamard
def H(): ...


# 量子回路を作る
ADD = ...
MUL = ...
SQR = ...
F = ...
F_1 = ...
QFT = ...
QFT_1 = ...


def add(): ...


def mul(): ...


def sqr(): ...


def f(): ...


def f_inverse(): ...


def qft(): ...


def qft_inverse(): ...


# T(・)で測定
def T(value: int, target_bit: int) -> int:
    return (value >> target_bit) & 1


# 1の個数を数える
def count(): ...


def main():
    start = time.time()
    constants = Constants()

    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.draw("mpl")
    plt.show()

    # points = create_single_point(constants)  # 四角形 # TODO - 分岐
    # hologram = generate_hologram(points, constants)
    # print("CGH Calculation completed!")

    # end = time.time()
    # print(print("Cal time:{} sec".format(end - start)))

    # print("Preparing for display...")
    # show(hologram)


if __name__ == "__main__":
    main()
