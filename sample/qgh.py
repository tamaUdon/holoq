# 点群法で量子コンピュータ生成ホログラムを作る

import time
import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
from constants import QuantumConstants
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import DraperQFTAdder, MultiplierGate, QFTGate, QFT, HGate


def init_superposition_state(qconsts: QuantumConstants) -> QuantumCircuit:
    xj_reg = QuantumRegister(qconsts.bits_w, "xj")
    xh_reg = QuantumRegister(qconsts.bits_w, "xh")
    anc1 = QuantumRegister(2, "anc1")  # |0>
    anc2 = QuantumRegister(2, "anc2")  # |0>
    yj_reg = QuantumRegister(qconsts.bits_w, "yj")
    yh_reg = QuantumRegister(qconsts.bits_w, "yh")
    anc3 = QuantumRegister(2, "anc3")  # |0>
    anc4 = QuantumRegister(2, "anc4")  # |0>
    rho_reg = QuantumRegister(qconsts.bits_w, "rho")
    anc5 = QuantumRegister(2, "anc5")  # |0>

    qc = QuantumCircuit(
        xj_reg, xh_reg, anc1, anc2, yj_reg, yh_reg, anc3, anc4, rho_reg, anc5
    )  # |xh>|0>|0>|yh>|0>|0>|ρj>|0>

    xj_offset = qc.find_bit(xj_reg[0]).index
    xh_offset = qc.find_bit(xh_reg[0]).index
    yj_offset = qc.find_bit(yj_reg[0]).index
    yh_offset = qc.find_bit(yh_reg[0]).index
    rho_offset = qc.find_bit(rho_reg[0]).index
    state = np.zeros(1 << qc.num_qubits, dtype=complex)

    def _float_to_int(value: float):
        """
        :bit幅に合わせて0.0-1.0を刻む補助関数:
        - 4bitの場合 0.3 -> 3 -> 0011にmapする
        """
        return round(value * ((1 << qconsts.bits_w) - 1))

    for j in range(qconsts.N):  # Σ
        xj, yj = qconsts.xj_yj[j]
        xh, yh = qconsts.xh_yh[j]  # 古典ビット
        rho = qconsts.ρ[j]
        basis_idx = (
            (_float_to_int(xj) << xj_offset)
            | (_float_to_int(xh) << xh_offset)
            | (_float_to_int(yj) << yj_offset)
            | (_float_to_int(yh) << yh_offset)
            | (_float_to_int(rho) << rho_offset)
        )
        state[basis_idx] += 1 / math.sqrt(qconsts.N)  # 1/√N
    qc.initialize(Statevector(state))
    return qc


def compose_circuits(
    qc: QuantumCircuit, num_state_qubits: int, qconstants: QuantumConstants
):
    """
    ### 量子回路を定義する関数
    :qc: 量子回路のインスタンス
    :num_state_qubits: 入力レジスタのビット数

    """
    # ゲートの定義
    qft = QFT(num_qubits=qconstants.N, insert_barriers=True)
    qft_1 = QFT(num_qubits=qconstants.N, inverse=True, insert_barriers=True)
    adder = DraperQFTAdder(
        num_state_qubits=num_state_qubits, kind="half"
    )  # TODO - [調査] halfとfixedの違い?
    mul = MultiplierGate(num_state_qubits=num_state_qubits, num_result_qubits=...)
    sqr = mul  # TODO - [実装]SQRを実装する

    # 入力レジスタの定義
    xj_reg, xh_reg, anc1, anc2, yj_reg, yh_reg, anc3, anc4, rho_reg, anc5 = qc.qregs
    # neg_xj = (1 << bits_w) - xj_i
    # neg_yj = (1 << bits_w) - yj_i # TODO - [実装]入力値のxj,yjを負数にする

    φ_0 = qft(0)  # TODO - [実装][調査]qft(0)の値を計算しておく

    # 量子回路の定義
    qc.append(adder, xh_reg - xj_reg)  # xh - xj # TODO [実装]負数の表現
    qc.append(adder, yh_reg - yj_reg)  # yh - yj # TODO [実装]結果をanc1,3に代入する
    qc.append(qft_1, anc1)  # QFT_1
    qc.append(qft_1, anc3)  # QFT_1 # TODO [実装]結果をanc1,3に代入する(そのまま)
    qc.append(mul, anc1 * φ_0)  # SQR # φ(xhj^2) # TODO [実装]anc1 * φ_0 (外積)
    qc.append(mul, anc3 * φ_0)  # SQR # φ(yhj^2) # TODO [実装]結果をanc2,**3**に代入する
    qc.append(qft_1, anc2)  # QFT_1
    qc.append(qft_1, anc3)  # QFT_1 # TODO [実装]結果をanc2,**3**に代入する(そのまま)
    qc.append(adder, anc2 + anc3)  # φ(xhj^2 + yhj^2) # TODO [実装]結果をanc4に代入する
    qc.append(qft_1, anc4)  # ρj # TODO [実装]結果をrho_regに代入する
    qc.append(mul, anc4, rho_reg)  # 𝜙(𝜌𝑗(𝑥𝑗ℎ2+𝑦𝑗ℎ2)) # TODO [実装]結果をanc5に代入する
    qc.append(qft_1, anc5)  # 𝜌𝑗(𝑥𝑗ℎ2+𝑦𝑗ℎ2) # TODO [実装]結果をanc5に代入する
    # -- ここまでで量子ホログラムが計算できている --#

    # TODO ここで anc5 に対して T(・)


def main():
    qconstants = QuantumConstants()
    qc = init_superposition_state(qconsts=qconstants)
    qc.decompose().draw("mpl")
    plt.show()


if __name__ == "__main__":
    main()

# qc.qregs
# [QuantumRegister(2, 'xj'), QuantumRegister(2, 'xh'),
# QuantumRegister(2, 'anc1'), QuantumRegister(2, 'yj'),
# QuantumRegister(2, 'yh'), QuantumRegister(2, 'anc2'),
#  QuantumRegister(2, 'rho'), QuantumRegister(1, 'anc3')]

# 1. 論文フロー通りに回路を連結する
#       xh−xj, yh−yj
#       xhj^2, yhj^2
#       xhj^2 + yhj^2
#       ρj * (xhj^2 + yhj^2)
#       ターゲットビットを抽出して測定する T() の処理
# 2. 4量子ビットのテストケースを実装する
#       論文にある値でok
#       手計算と合うか確認する
# 3. 測定結果の確認
#       3点の場合をテストする
#       古典計算、手計算の値と合うか確認する
