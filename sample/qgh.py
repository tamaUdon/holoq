# 点群法で量子コンピュータ生成ホログラムを作る

import time
import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
from constants import QuantumConstants
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import (
    DraperQFTAdder,
    RGQFTMultiplier,
)


def _add_bitw(x):
    # 加算のレジスタ幅
    return x + 1


def _mul_bitw(x, y):
    # 乗算のレジスタ幅
    return x + y


def define_regs(qconsts: QuantumConstants, test: bool) -> QuantumCircuit:
    # レジスタの定義
    base = qconsts.bits_w  # 2
    add_w = _add_bitw(base)  # 3
    mul_w = _mul_bitw(add_w, add_w)  # 6
    sq_w = _add_bitw(mul_w)  # 7
    res_w = _add_bitw(sq_w)  # 8

    xj_reg = QuantumRegister(base, "xj")
    xh_reg = QuantumRegister(base, "xh")
    xhj_reg = AncillaRegister(add_w, "xhj")
    xhj_b_reg = AncillaRegister(add_w, "xhj_b")
    xhj_sq_reg = AncillaRegister(mul_w, "xhj_sq_reg")
    yj_reg = QuantumRegister(qconsts.bits_w, "yj")
    yh_reg = QuantumRegister(qconsts.bits_w, "yh")
    yhj_reg = AncillaRegister(add_w, "yhj")
    yhj_b_reg = AncillaRegister(add_w, "yhj_b")
    yhj_sq_reg = AncillaRegister(sq_w, "yhj_sq_reg")
    rho_reg = QuantumRegister(sq_w, "rho")
    result = AncillaRegister(res_w, "result")
    cl_result = ClassicalRegister(res_w, "cl_result")

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
    mul = RGQFTMultiplier(num_state_qubits=7, num_result_qubits=8)
    sqr = RGQFTMultiplier(
        num_state_qubits=3,
        num_result_qubits=6,
        name="SQR_RGQFTMultiplier",
    )
    return adder, adder_sum, mul, sqr


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
        # circuit.x(xh_reg[0])  # |01>
        # circuit.x(yh_reg[0])  # |01>
        circuit.x(xj_reg[0])  # |01>
        circuit.x(yj_reg[0])  # |01>
        circuit.x(rho_reg[0])  # |1>
    else:
        xj_offset = circuit.find_bit(xj_reg[0]).index
        xh_offset = circuit.find_bit(xh_reg[0]).index
        yj_offset = circuit.find_bit(yj_reg[0]).index
        yh_offset = circuit.find_bit(yh_reg[0]).index
        rho_offset = circuit.find_bit(rho_reg[0]).index
        state = np.zeros(
            1 << circuit.num_qubits, dtype=complex
        )  # WARINIG - Memory Error! Numpy try to allocate 4.00 PiB...

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


def compose_circuits(
    circuit: QuantumCircuit, qgates: tuple, test: bool
) -> QuantumCircuit:
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
    adder, adder_sum, mul, sqr = qgates

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

    # ② SQR
    circuit.append(sqr, list(xhj_reg) + list(xhj_sub_reg) + list(xhj_sq_reg))
    circuit.append(sqr, list(yhj_reg) + list(yhj_sub_reg) + list(yhj_sq_reg[:6]))

    # ③ ADD
    circuit.append(adder_sum, list(xhj_sq_reg) + list(yhj_sq_reg))

    # ④ MUL
    circuit.append(mul, list(yhj_sq_reg) + list(rho_reg) + list(result_reg))

    # MEASURE
    if test:
        circuit.measure_all()  # 全てのビットを確認するとき
    else:
        circuit.measure(qubit=result_reg[0], cbit=cl_result[0])  # T(・) 上位ビットのみ
    return circuit


def execute(circuit: QuantumCircuit):
    """
    ### 回路をシミュレートする関数

    :param circuit: 量子回路のインスタンス
    :param type: QuantumCircuit
    """
    simulator = AerSimulator(
        method="matrix_product_state"
    )  # MEMO - StateVectorで検証すると動かなかったのでMPSで試した

    transpiled_circuit = transpile(
        circuit,
        simulator,
        coupling_map=None,  # WARNING - ロジック検証用, ハードウェアの仮定なし, 実機でのデバッグ時は指定必須
        optimization_level=1,
    )

    job = simulator.run(transpiled_circuit, shots=6)
    result = job.result()
    counts = result.get_counts(circuit)  # qubit = anc5[0]をカウント

    return counts


def main():
    TEST = True
    qconstants = QuantumConstants()
    gates = define_gates()
    circuit = define_regs(qconsts=qconstants, test=TEST)
    circuit = init_superposition_state(circuit=circuit, qconsts=qconstants, test=TEST)
    circuit = compose_circuits(circuit=circuit, qgates=gates, test=TEST)

    print(circuit.draw("text"))

    start = time.time()
    counts = execute(circuit=circuit)
    end = time.time()

    print(f" Execution took {end - start} seconds.")

    print(f"{counts=}")

    integer_counts = {}
    for binary_string, count in counts.items():
        print(f"{binary_string=}")  # 1,0のような文字列が入っている
        integer_value = int(binary_string[0], 2)
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
# 1. 測定結果の確認
#       古典計算、手計算の値と合うか確認する
# 2. MULのextentionを作成しSQRにする
# 3. デコレーターの実装. 関数の前後にログを出力する関数を作る
