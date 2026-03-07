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
from qiskit.circuit.library import (
    DraperQFTAdder,
    MultiplierGate,
    QFT,
    HGate,
    RGQFTMultiplier,
)


def init_superposition_state(qconsts: QuantumConstants) -> QuantumCircuit:
    xj_reg = QuantumRegister(qconsts.bits_w, "xj")
    xh_reg = QuantumRegister(qconsts.bits_w, "xh")
    anc1 = QuantumRegister(1, "anc1")  # |0>
    anc2 = QuantumRegister(qconsts.bits_w * 2, "anc2")  # |0>
    yj_reg = QuantumRegister(qconsts.bits_w, "yj")
    yh_reg = QuantumRegister(qconsts.bits_w, "yh")
    anc3 = QuantumRegister(1, "anc3")  # |0>
    anc4 = QuantumRegister(qconsts.bits_w * 2, "anc4")  # |0>
    rho_reg = QuantumRegister(qconsts.bits_w, "rho")
    anc5 = QuantumRegister(qconsts.bits_w, "anc5")  # |0>

    print("init reg")

    qc = QuantumCircuit(
        xj_reg, xh_reg, anc1, anc2, yj_reg, yh_reg, anc3, anc4, rho_reg, anc5
    )  # |xh>|0>|0>|yh>|0>|0>|ρj>|0>

    xj_offset = qc.find_bit(xj_reg[0]).index
    xh_offset = qc.find_bit(xh_reg[0]).index
    yj_offset = qc.find_bit(yj_reg[0]).index
    yh_offset = qc.find_bit(yh_reg[0]).index
    rho_offset = qc.find_bit(rho_reg[0]).index
    state = np.zeros(1 << qc.num_qubits, dtype=complex)

    print("calc offset")

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

    print("calc basis index")

    qc.initialize(Statevector(state))  # took long time
    # TODO - qcの内容をhdf5などに一時保存する方法を調べる
    return qc


def compose_circuits(qc: QuantumCircuit, qconsts: QuantumConstants) -> QuantumCircuit:
    """
    ### 量子回路を定義する関数
    :qc: 量子回路のインスタンス
    :num_state_qubits: 入力レジスタのビット数
    """

    # 入力レジスタの定義
    xj_reg, xh_reg, anc1, anc2, yj_reg, yh_reg, anc3, anc4, rho_reg, anc5 = qc.qregs
    # neg_xj = (1 << bits_w) - xj_i
    # neg_yj = (1 << bits_w) - yj_i # TODO - [実装]入力値のxj,yjを負数にする

    # ゲートの定義
    qft = QFT(num_qubits=qconsts.bits_w, insert_barriers=True)
    qft_1 = QFT(num_qubits=1, inverse=True, insert_barriers=True)
    adder = DraperQFTAdder(num_state_qubits=qconsts.bits_w, kind="half")
    mul = RGQFTMultiplier(
        num_state_qubits=qconsts.bits_w, num_result_qubits=2 * qconsts.bits_w
    )
    sqr = RGQFTMultiplier(
        num_state_qubits=qconsts.bits_w,
        num_result_qubits=2 * qconsts.bits_w,
        name="SQR_RGQFTMultiplier",
    )
    print(f"{qft_1=}")
    print(f"{adder=}")
    print(f"{mul=}")
    print(f"{sqr=}")

    # 量子回路の定義
    qc.append(
        adder, list(xh_reg) + list(xj_reg) + list(anc1)
    )  # xh - xj # TODO [実装]負数の表現
    qc.append(
        adder, list(yh_reg) + list(yj_reg) + list(anc3)
    )  # yh - yj # TODO [実装]結果をanc1,3に代入する
    qc.append(qft_1, anc1)  # QFT_1
    qc.append(qft_1, anc3)  # QFT_1 # TODO [実装]結果をanc1,3に代入する(そのまま)

    scratch_reg = QuantumRegister(qconsts.bits_w * 2, "scratch")
    qc.add_register(scratch_reg)

    # for i in range(num_state_qubits):
    #     qc.cx(anc1[i], scratch_reg[i])  # |0> -> |anc1>にコピー
    qc.append(
        sqr, list(anc1) + list(anc2), copy=True
    )  # SQR ... |a⟩|b⟩|0⟩ → |a⟩|b⟩|a×b⟩ # FIXME - qiskit.circuit.exceptions.CircuitError: 'The amount of qubit arguments 5 does not match the instruction expectation (8).'
    # SQR # φ(xhj^2) # TODO [実装]anc1 * φ_0 (外積)
    # SQR # φ(yhj^2) # TODO [実装]結果をanc2,**3**に代入する

    qc.append(qft_1, anc2)  # QFT_1
    qc.append(qft_1, anc3)  # QFT_1 # TODO [実装]結果をanc2,**3**に代入する(そのまま)

    qc.append(
        adder, list(anc2) + list(anc3)
    )  # φ(xhj^2 + yhj^2) # TODO [実装]結果をanc4に代入する
    qc.append(qft_1, anc4)  # ρj # TODO [実装]結果をrho_regに代入する
    qc.append(
        mul, list(anc4) + list(rho_reg) + list(anc5)
    )  # 𝜙(𝜌𝑗(𝑥𝑗ℎ2+𝑦𝑗ℎ2)) # TODO [実装]結果をanc5に代入する
    qc.append(qft_1, anc5)  # 𝜌𝑗(𝑥𝑗ℎ2+𝑦𝑗ℎ2) # TODO [実装]結果をanc5に代入する
    # -- ここまでで量子ホログラムが計算できている --#
    # TODO ここで anc5 に対して T(・)

    return qc


def main():
    qconstants = QuantumConstants()
    qc = init_superposition_state(qconsts=qconstants)
    print("qc is initialized")
    qc = compose_circuits(qc=qc, qconsts=qconstants)
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
