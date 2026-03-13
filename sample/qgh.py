# 点群法で量子コンピュータ生成ホログラムを作る

import time
import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
from constants import QuantumConstants
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit_aer import AerSimulator, Aer
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit.quantum_info import Statevector
from qiskit.circuit.library import (
    DraperQFTAdder,
    MultiplierGate,
    QFT,
    HGate,
    RGQFTMultiplier,
)

TEST = True


def define_regs(qconsts: QuantumConstants, test=False) -> QuantumCircuit:
    # レジスタの定義
    xj_reg = QuantumRegister(qconsts.bits_w, "xj")
    xh_reg = QuantumRegister(qconsts.bits_w, "xh")
    xhj_reg = AncillaRegister(3, "xhj")
    xhj_b_reg = AncillaRegister(3, "xhj_b")
    xhj_sq_reg = AncillaRegister(6, "xhj_sq_reg")
    yj_reg = QuantumRegister(qconsts.bits_w, "yj")
    yh_reg = QuantumRegister(qconsts.bits_w, "yh")
    yhj_reg = AncillaRegister(3, "yhj")  # 6bitでよいが最後のADDに合わせて7bitにする
    yhj_b_reg = AncillaRegister(3, "yhj_b")  # 6bitでよいが最後のADDに合わせて7bitにする
    yhj_sq_reg = AncillaRegister(7, "yhj_sq_reg")
    rho_reg = QuantumRegister(7, "rho")  # 1bitでよいがanc4と合わせる
    result = AncillaRegister(8, "result")
    cl_result = None

    if test:
        cl_result = ClassicalRegister(8, "cl_full")
    else:
        cl_result = ClassicalRegister(1, "cl_result")

    qc = QuantumCircuit(
        xj_reg,
        xh_reg,
        xhj_reg,
        xhj_b_reg,
        xhj_sq_reg,
        yj_reg,
        yh_reg,
        yhj_reg,
        yhj_b_reg,
        yhj_sq_reg,
        rho_reg,
        result,
        cl_result,
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
    circuit: QuantumCircuit, qconsts: QuantumConstants, test=False
) -> QuantumCircuit:
    (
        xj_reg,
        xh_reg,
        _,
        _,
        _,
        yj_reg,
        yh_reg,
        _,
        _,
        _,
        rho_reg,
        _,
    ) = circuit.qregs

    if test:
        circuit.x(xh_reg[0])  # |01>
        circuit.x(yh_reg[0])  # |01>
        circuit.x(rho_reg[0])  # |1>
    else:
        xj_offset = circuit.find_bit(xj_reg[0]).index
        xh_offset = circuit.find_bit(xh_reg[0]).index
        yj_offset = circuit.find_bit(yj_reg[0]).index
        yh_offset = circuit.find_bit(yh_reg[0]).index
        rho_offset = circuit.find_bit(rho_reg[0]).index
        state = np.zeros(1 << circuit.num_qubits, dtype=complex)

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

        circuit.initialize(Statevector(state))  # took long time
    return circuit


def compose_circuits(circuit: QuantumCircuit, qgates: tuple) -> QuantumCircuit:
    """
    ### 量子回路を定義する関数
    :circuit: 量子回路のインスタンス
    :num_state_qubits: 入力レジスタのビット数
    """

    (
        xj_reg,
        xh_reg,
        xhj_reg,
        xhj_sub_reg,
        xhj_sq_reg,
        yj_reg,
        yh_reg,
        yhj_reg,
        yhj_sub_reg,
        yhj_sq_reg,
        rho_reg,
        result_reg,
    ) = circuit.qregs
    cl_result = circuit.cregs
    adder, adder_sum, qft_1, qft_2_3, qft_4, qft_5, mul, sqr = qgates

    # ⓪ Copy from xj to xhj, yj to yhj
    for n in range(len(xj_reg)):
        circuit.cx(xj_reg[n], xhj_reg[n])
        circuit.cx(yj_reg[n], yhj_reg[n])  # ビット数はyh_reg = xj_regとする

    # ① ADD.inverse
    circuit.append(adder.inverse(), list(xh_reg) + list(xhj_reg))
    circuit.append(adder.inverse(), list(yh_reg) + list(yhj_reg))

    # ①' Copy from xhj to xhj_b, yhj tp yhj_b
    for n in range(len(xhj_reg)):
        circuit.cx(xhj_reg[n], xhj_sub_reg[n])
        circuit.cx(yhj_reg[n], yhj_sub_reg[n])

    # ①'' QFT_1
    circuit.append(qft_1, xhj_sub_reg)
    circuit.append(qft_1, yhj_sub_reg)

    # ② SQR
    circuit.append(sqr, list(xhj_reg) + list(xhj_sub_reg) + list(xhj_sq_reg))
    circuit.append(sqr, list(yhj_reg) + list(yhj_sub_reg) + list(yhj_sq_reg[:6]))

    # ②'' QFT_1
    circuit.append(qft_2_3, xhj_sq_reg)
    circuit.append(qft_2_3, yhj_sq_reg[:6])

    # ③ ADD
    circuit.append(adder_sum, list(xhj_sq_reg) + list(yhj_sq_reg))

    # ③' QFT_1
    circuit.append(qft_4, yhj_sq_reg)

    # ④ MUL
    circuit.append(mul, list(yhj_sq_reg) + list(rho_reg) + list(result_reg))

    # ④' QFT_1
    circuit.append(qft_5, result_reg)

    # MEASURE
    if TEST:
        circuit.measure_all()
    else:
        circuit.measure(qubit=result_reg[0], cbit=cl_result[0])
    return circuit


def execute(circuit: QuantumCircuit):
    # 回路をシミュレート
    simulator = AerSimulator(
        method="matrix_product_state"
    )  # MEMO - StateVectorで検証すると動かなかったのでMPSで試した
    # MEMO - n_qbits=49: 2^49 × 16 bytes ... 8 PiB
    transpiled_circuit = transpile(
        circuit,
        simulator,
        coupling_map=None,  # WARNING - ロジック検証用, ハードウェアの仮定なし, 実機でのデバッグ時は指定必須
        optimization_level=3,  # 回路が大きいので最適化レベルを最高値に設定
    )
    job = simulator.run(transpiled_circuit, shots=6)
    result = job.result()
    counts = result.get_counts(circuit)  # qubit = anc5[0]をカウント

    return counts


def main():
    qconstants = QuantumConstants()
    gates = define_gates()
    circuit = define_regs(qconsts=qconstants, test=TEST)
    circuit = init_superposition_state(circuit=circuit, qconsts=qconstants, test=TEST)
    circuit = compose_circuits(circuit=circuit, qgates=gates)

    print(circuit.draw("text"))

    start = time.time()
    counts = execute(circuit=circuit)
    end = time.time()

    print(f" Execution took {end - start} seconds.")

    if TEST:
        print(f"{counts=}")
    else:
        integer_counts = {}
        for binary_string, count in counts.items():
            print(f"{binary_string=}")  # 1,0のような文字列が入っている
            integer_value = int(binary_string, 2)
            integer_counts[integer_value] = count

        print(f"Measurement counts (binary strings): {counts}")
        print(f"Measurement counts (integers): {integer_counts}")

        plt.bar(list(integer_counts.keys()), list(integer_counts.values()))
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.title("Measurement result")
        plt.xticks(list(integer_counts.keys()))
        plt.show()


if __name__ == "__main__":
    main()

# 残り TODO
# 1. anc1-5を自動的に決定する関数を作成する
# 2. 測定結果の確認
#       古典計算、手計算の値と合うか確認する
# 3. MULのextentionを作成しSQRにする
# 4. デコレーターの実装. 関数の前後にログを出力する関数を作る
