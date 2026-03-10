# 点群法で量子コンピュータ生成ホログラムを作る

import time
import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
from constants import QuantumConstants
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import (
    DraperQFTAdder,
    MultiplierGate,
    QFT,
    HGate,
    RGQFTMultiplier,
)


def define_regs(qconsts: QuantumConstants) -> QuantumCircuit:
    # レジスタの定義
    xj_reg = QuantumRegister(qconsts.bits_w, "xj")
    xh_reg = QuantumRegister(qconsts.bits_w, "xh")
    anc1 = QuantumRegister(6, "anc1")  # 3bitでよいがSQRに合わせて6bitにする
    anc2 = QuantumRegister(6, "anc2")
    yj_reg = QuantumRegister(qconsts.bits_w, "yj")
    yh_reg = QuantumRegister(qconsts.bits_w, "yh")
    anc3 = QuantumRegister(7, "anc3")  # 6bitでよいが最後のADDに合わせて7bitにする
    anc4 = QuantumRegister(7, "anc4")
    rho_reg = QuantumRegister(7, "rho")  # 1bitでよいがanc4と合わせる
    anc5 = QuantumRegister(8, "anc5")
    cl1 = ClassicalRegister(1, "cl1")

    qc = QuantumCircuit(
        xj_reg, xh_reg, anc1, anc2, yj_reg, yh_reg, anc3, anc4, rho_reg, anc5, cl1
    )
    return qc


def define_gates() -> tuple:
    # ゲートの定義
    adder = DraperQFTAdder(num_state_qubits=2, kind="half")
    adder_sum = DraperQFTAdder(num_state_qubits=6, kind="half")
    qft_1 = QFT(num_qubits=3, inverse=True, insert_barriers=True)
    qft_2_3 = QFT(num_qubits=6, inverse=True, insert_barriers=True)
    qft_4 = QFT(num_qubits=7, inverse=True, insert_barriers=True)
    qft_5 = QFT(num_qubits=8, inverse=True, insert_barriers=True)
    mul = RGQFTMultiplier(num_state_qubits=7, num_result_qubits=8)
    sqr = RGQFTMultiplier(
        num_state_qubits=3,
        num_result_qubits=6,
        name="SQR_RGQFTMultiplier",
    )
    return adder, adder_sum, qft_1, qft_2_3, qft_4, qft_5, mul, sqr


def init_superposition_state(
    qc: QuantumCircuit, qconsts: QuantumConstants, test=False
) -> QuantumCircuit:
    xj_reg, xh_reg, _, _, yj_reg, yh_reg, _, _, rho_reg, _ = qc.qregs

    if test:
        qc.x(xh_reg[0])  # |01>
        qc.x(yh_reg[0])  # |01>
        qc.x(rho_reg[0])  # |1>
    else:
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
    return qc


def compose_circuits(qc: QuantumCircuit, gates: tuple, N: int) -> QuantumCircuit:
    """
    ### 量子回路を定義する関数
    :qc: 量子回路のインスタンス
    :num_state_qubits: 入力レジスタのビット数
    """

    xj_reg, xh_reg, anc1, anc2, yj_reg, yh_reg, anc3, anc4, rho_reg, anc5 = qc.qregs
    cl1 = qc.clbits
    adder, adder_sum, qft_1, qft_2_3, qft_4, qft_5, mul, sqr = gates
    # TODO - [実装]入力値のxj,yjを負数にする

    # ADD -> QFT_1
    for n in range(len(xj_reg)):  # xj_regの長さ分
        qc.cx(xj_reg[n], anc1[n])  # xj -> anc1にコピー
        qc.cx(yj_reg[n], anc3[n])  # ビット数はyh_reg = xj_regの前提
    qc.append(adder, list(xh_reg) + list(anc1)[:3])
    qc.append(adder, list(yh_reg) + list(anc3)[:3])  # コピーした3つ分のみ取り出す
    qc.append(qft_1, anc1[:3])
    qc.append(qft_1, anc3[:3])

    # SQR -> QFT_1
    qc.append(sqr, list(anc1) + list(anc2))
    qc.append(sqr, list(anc3[:6]) + list(anc4[:6]))
    qc.reset(anc3)
    for n in range(len(anc3[:6])):
        qc.cx(anc4[n], anc3[n])  # anc4 -> anc3にコピー
    qc.append(qft_2_3, anc2)
    qc.append(qft_2_3, anc3[:6])
    qc.reset(anc4)  # anc4を次のADDのために空ける

    # ADD -> QFT_1
    qc.append(adder_sum, list(anc2) + list(anc3))
    qc.cx(anc3, anc4)  # anc3をanc4にコピー
    qc.append(qft_4, anc4)

    # MUL -> QFT_1
    qc.append(mul, list(anc4) + list(rho_reg) + list(anc5))
    qc.append(qft_5, anc5)

    # 物体点数回 T(・)
    counts = 0
    for n in range(N):
        qc.measure(qubit=anc5[0], cbit=cl1)  # TODO - result()? measure()?
        # counts += (  # レジスタの1桁目が1の場合を数える
        #     1 if int(cl1) == 1 else 0
        # )  # TODO - int(cl1)はできない. 古典ビットに入った値を0か1で表現するには?
        # TODO - cl1がずっと1なのが気になる
        print(f"{cl1=}")
        print(f"{counts=}")

    return qc


def main():
    qconstants = QuantumConstants()
    gates = define_gates()
    qc = define_regs(qconsts=qconstants)
    qc = init_superposition_state(qc=qc, qconsts=qconstants, test=True)
    print("qc is initialized")
    qc = compose_circuits(qc=qc, gates=gates, N=qconstants.N)
    print(qc.draw("text"))  # 回路が巨大すぎて描画できないのでtext形式で出力する


if __name__ == "__main__":
    main()

# 残り TODO
# 1. anc1-5を自動的に決定する関数を作成する

# 実装順
# 1. ターゲットビットを抽出して測定する T() の処理
# 2. 測定結果の確認
#       3点の場合をテストする
#       古典計算、手計算の値と合うか確認する
