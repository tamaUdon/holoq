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


def define_regs(): ...


def define_gates(): ...


def init_superposition_state(qconsts: QuantumConstants) -> QuantumCircuit:
    xj_reg = QuantumRegister(qconsts.bits_w, "xj")
    xh_reg = QuantumRegister(qconsts.bits_w, "xh")
    anc1 = QuantumRegister(3, "anc1")
    anc2 = QuantumRegister(6, "anc2")
    yj_reg = QuantumRegister(qconsts.bits_w, "yj")
    yh_reg = QuantumRegister(qconsts.bits_w, "yh")
    anc3 = QuantumRegister(6, "anc3")
    anc4 = QuantumRegister(7, "anc4")
    rho_reg = QuantumRegister(qconsts.bits_w, "rho")
    anc5 = QuantumRegister(10, "anc5")

    print("init reg")

    qc = QuantumCircuit(
        xj_reg, xh_reg, anc1, anc2, yj_reg, yh_reg, anc3, anc4, rho_reg, anc5
    )

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
    qft_1 = QFT(num_qubits=3, inverse=True, insert_barriers=True)
    qft_3 = QFT(num_qubits=6, inverse=True, insert_barriers=True)
    adder = DraperQFTAdder(num_state_qubits=6, kind="half")
    adder_3 = DraperQFTAdder(num_state_qubits=12, kind="half")
    mul = RGQFTMultiplier(num_state_qubits=qconsts.bits_w, num_result_qubits=7)
    sqr = RGQFTMultiplier(
        num_state_qubits=12,
        name="SQR_RGQFTMultiplier",
    )

    # 量子回路の定義
    qc.append(adder, list(xh_reg) + list(xj_reg) + list(anc1))
    qc.append(adder, list(yh_reg) + list(yj_reg) + list(anc3))
    qc.append(qft_1, anc1)
    qc.append(qft_1, anc3)
    qc.append(sqr, list(anc1) + list(anc2))
    qc.append(
        sqr,
        list(anc3),
    )
    qc.append(qft_3, anc2)
    qc.append(qft_1, anc3)
    qc.append(adder_3, list(anc2) + list(anc3) + list(anc4))
    qc.append(qft_3, anc4)
    qc.append(mul, list(anc4) + list(rho_reg) + list(anc5))
    qc.append(qft_3, anc5)
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

# 残り TODO
# 1. anc1-5を自動的に決定する関数を作成する

# 実装順
# 1. ターゲットビットを抽出して測定する T() の処理
# 2. 4量子ビットのテストケースを実装する
#       論文にある値でok
#       手計算と合うか確認する
# 3. 測定結果の確認
#       3点の場合をテストする
#       古典計算、手計算の値と合うか確認する
