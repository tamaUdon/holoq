# 点群法で量子コンピュータ生成ホログラムを作る

import time
import math
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Iterable, Tuple
from constants import Constants
from pointcloud import create_single_point, generate_hologram, show
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import DraperQFTAdder
class QRegister:
    def __init__(self, n) -> None:
        self.n = n
        self.ψ = np.zeros((2,) * n)  # 初期化
        self.ψ[(0,) * n] = 1  # ψ[0,0,...,0]を1に置き換える
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

MUL = ...
SQR = ...
F = ...
F_1 = ...
QFT = ...
QFT_1 = ...


def add():
    adder= DraperQFTAdder(3,kind="fixed")
    print(adder.decompose().draw())



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
    
    add()

    # points = create_single_point(constants)  # 四角形 # TODO - 分岐
    # hologram = generate_hologram(points, constants)
    # print("CGH Calculation completed!")

    #print("Preparing for display...")
    #show(hologram)


if __name__ == "__main__":
    main()

# 0. 量子コンピュータの頭の中 p171を写経（基本的な使い方を確認）
# 1. 量子コンピュータの頭の中 p246を写経（量子回路~ノイズモデル~測定を確認）
# 2. p171を4量子ビットに拡張
# 3. QFT加算器をQiskitで作る方法を調べる -> ok
#       実装する
# 4. QFT乗算器をQiskitで作る方法を調べる
#       実装する
# 5. QFTSQRをQiskitで作る方法を調べる
#       実装する
# 6. QFT, QFT-1をQiskitで実装する方法を調べる
#       実装する
# 7. QFTの単体テストをする
# 8. 規定エンコードを実装する
#       論文のエンコード部分を読む
#       作り方を調べる
#       実装する
# 9. ρj と座標のスケール係数（scale）と target_bitを決める
#       実装する
# 10. ρjの単体テストをする
# 11. 論文フロー通りに回路を連結する
#       xh−xj, yh−yj（2の補数で負数を表現）
#       xhj^2, yhj^2
#       xhj^2 + yhj^2
#       ρj * (xhj^2 + yhj^2)
#       ターゲットビットを抽出して測定する T() の処理
# 11. 4量子ビットのテストケースを実装する
#       論文にある値でok
#       手計算と合うか確認する
# 12. 測定結果の確認
#       3点の場合をテストする
#       古典計算の値と大体合うか確認する
