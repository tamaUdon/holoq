# 点群法で量子コンピュータ生成ホログラムを作る

import time
import math
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from constants import Constants
from pointcloud import create_single_point, generate_hologram, show
from qiskit import QuantumCircuit, QuantumRegister

class QRegister:
    def __init__(self, n) -> None:
        self.n = n
        self.ψ = np.zeros((2,) * n)  # 初期化
        self.ψ[(0,) * n] = 1  # ψ[0,0,...,0]を1に置き換える


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
        H_matrix()
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
H_matrix = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

# Controlled-Notゲート
# 標的ビットを制御するゲート
CNOT_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
# 2量子ビット版
CNOT_tensor = np.reshape(CNOT_matrix, (2, 2, 2, 2))


# Controlled-NOT
def CNOT(control: int, target: int, reg: QRegister) -> QRegister:
    # def H の一般化, n量子ビット対応
    ...


# Hadamard
def H(i, reg: QRegister) -> QRegister:
    # アダマールゲートを作用させる関数
    ...


# 量子回路を作る
def ADD():
    # 2量子ビット+2古典ビットの回路
    qc = QuantumCircuit(2,2)
    qc.h(0) # Hadamard gate
    qc.cx(0,1) # CNOT -> 0,1番目の量子ビット間に追加
    m = qc.measure_all() # 測定
    print("measured", m)

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
def T(): ...


# 1の個数を数える
def count(): ...


def main():
    start = time.time()
    constants = Constants()

    #points = create_single_point(constants)  # 四角形 # TODO - 分岐
    #hologram = generate_hologram(points, constants)
    #print("CGH Calculation completed!")
    ADD()

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))

    #print("Preparing for display...")
    #show(hologram)


if __name__ == "__main__":
    main()
