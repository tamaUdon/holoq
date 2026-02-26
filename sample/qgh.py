# 点群法で量子コンピュータ生成ホログラムを作る

import time
import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Tuple
from constants import Constants
from constants import Constants

# Qiskit (Aer)
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import DraperQFTAdder, MultiplierGate, QFTGate, QFT

bits_width = Constants.bits_w


# 固定値
N = 3
a = [1, 2, 3, 0]
ρ = [0.5, 0.25, 0.5, 0]
xj_yj = [(0, 0), (1, 0), (0, 1), (1, 1)]
xh_yh = [(0, 0), (1, 0), (0, 1), (1, 1)]


def init_superposition_state():
    # xh_start = 0  # 0-index
    # xh_end = xh_start + bits_width  # |xh>
    # yh_start = xh_end + 2  # |0>|0>
    # yh_end = yh_start + bits_width  # |yh>
    # rho_start = yh_end + 2  # |0>|0>
    # total_bits_w = rho_start + bits_width + 1  # |ρj>|0>

    xh_reg = QuantumRegister(bits_width, "xh")
    anc1 = QuantumRegister(2, "anc1")  # |0>|0>
    yh_reg = QuantumRegister(bits_width, "yh")
    anc2 = QuantumRegister(2, "anc2")  # |0>|0>
    rho_reg = QuantumRegister(bits_width, "rho")
    anc3 = QuantumRegister(1, "anc3")  # |0>

    qc = QuantumCircuit(xh_reg, anc1, yh_reg, anc2, rho_reg, anc3)

    sub = prepare_basis_state(bits_width, xh_value)
    qc.compose(sub, qubits=xh_reg[:], inplace=True)

    # 値を入れたいレジスタ（xh, yh, rho）だけ prepare_basis_state() でXゲートをかける

    ...


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

# 1. 物体点数が1の場合で実装する
# 3. QFT加算器をQiskitで作る方法を調べる、手計算 -> ok
#       実装する
# 4. QFT乗算器をQiskitで作る方法を調べる、手計算 -> ok
#       実装する
# 5. QFTSQRをQiskitで作る方法を調べる、手計算 -> ok
#       実装する
# 6. QFT, QFT-1をQiskitで実装する方法を調べる、手計算 -> ok
#       実装する
# 7. QFTの単体テストをする
# 8. 基底エンコードを実装する
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
