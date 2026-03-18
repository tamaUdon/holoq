# 点群法で量子コンピュータ生成ホログラムを作る

import time
import math
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from constants import QuantumConstants, ClassicalConstants
from pointcloud import create_single_point
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit.circuit import QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import (
    DraperQFTAdder,
    RGQFTMultiplier,
)


def define_gates() -> tuple:
    """
    ゲートの定義
    """
    adder = DraperQFTAdder(num_state_qubits=2, kind="half")
    adder_sum = DraperQFTAdder(num_state_qubits=6, kind="half")
    mul = RGQFTMultiplier(num_state_qubits=7, num_result_qubits=8)
    sqr = RGQFTMultiplier(
        num_state_qubits=3,
        num_result_qubits=6,
        name="SQR_RGQFTMultiplier",
    )
    return adder, adder_sum, mul, sqr


def define_regs(qconsts: QuantumConstants) -> QuantumCircuit:
    """
    レジスタの定義
    """

    def _add_bitw(x):
        # 加算のレジスタ幅
        return x + 1

    def _mul_bitw(x, y):
        # 乗算のレジスタ幅
        return x + y

    base_w = qconsts.b_width  # 例...2のとき
    add_w = _add_bitw(base_w)  # 3
    mul_w = _mul_bitw(add_w, add_w)  # 6
    sq_w = _add_bitw(mul_w)  # 7
    res_w = _add_bitw(sq_w)  # 8

    xj_reg = QuantumRegister(base_w, "xj")
    xh_reg = QuantumRegister(base_w, "xh")
    xhj_reg = AncillaRegister(add_w, "xhj")
    xhj_b_reg = AncillaRegister(add_w, "xhj_b")
    xhj_sq_reg = AncillaRegister(mul_w, "xhj_sq_reg")
    yj_reg = QuantumRegister(base_w, "yj")
    yh_reg = QuantumRegister(base_w, "yh")
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


def init_superposition_state(
    circuit: QuantumCircuit, consts: QuantumConstants, test=False
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

    # 1. 重ね合わせを作る
    circuit.h(xj_reg)
    circuit.h(yj_reg)

    xj = [0, 1, 0, 1]
    yj = [0, 0, 1, 1]
    rho_j = [
        (1, 0),  # 0.5
        (0, 1),  # 0.25
        (1, 0),  # 0.5
        (0, 0),  # 0
    ]  # [0.5, 0.25, 0.5, 0]を二進数に変換 0|1...00, 0.25...01, 0.5...10, 0.75...11
    # a_j = [1,2,3,0] # -> aは一旦無視して計算する
    # xj_yj = [(0, 0), (1, 0), (0, 1), (1, 1)]  |00>, |01>, |10>, |11>として埋め込む

    # 2. 古典値のリストから値を入れる
    for i in range(consts.N):
        with circuit.if_test((xj[i], 1)):
            circuit.x(xj_reg[i])
        with circuit.if_test((yj[i], 1)):
            circuit.x(yj_reg[i])
        # with circuit.if_test((xj_reg[i], 1) or (yj_reg[i], 1)):
        #     # xj, yjに値が入っているならrhoにも値を入れる
        # TODO - offsetが必要かも
        for rho in rho_j:  # rho=(1,0)
            circuit.cx(control_qubit=rho[0], target_qubit=rho_reg[2 * i])
            circuit.cx(control_qubit=rho[1], target_qubit=rho_reg[2 * i + 1])
    print(circuit.draw())
    return circuit


def define_circuit(
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

    # 測定前にbarrier
    circuit.barrier()

    # MEASURE
    if test:
        circuit.measure_all()  # 全てのビットを確認するとき
    else:
        circuit.measure(qubit=result_reg[0], cbit=cl_result[0])  # T(・) 上位ビットのみ
    return circuit


def execute(circuit: QuantumCircuit) -> int:
    ...
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
    plot_histogram(counts)
    T = ...  # countsからTを取り出す

    return 0


def build_circuit(constants: QuantumConstants, gates: tuple) -> QuantumCircuit:
    circuit = define_regs(qconsts=constants)
    circuit = init_superposition_state(
        circuit=circuit, consts=constants, test=constants.TEST
    )
    circuit = define_circuit(circuit=circuit, qgates=gates, test=constants.TEST)
    print(circuit.draw("text"))
    return circuit


def generate_hologram_q(points: np.ndarray, qconstants: QuantumConstants) -> np.ndarray:
    gates = define_gates()
    hologram = np.zeros((qconstants.Y, qconstants.X), dtype=int)

    for xh in tqdm.tqdm(range(qconstants.X)):
        for yh in range(qconstants.Y):
            circuit = build_circuit(
                constants=qconstants, gates=gates
            )  # 測定ごとに回路を作り直す?
            T = execute(circuit=circuit)
            hologram[xh, yh] = T
    return hologram


def show(hologram: np.ndarray) -> None:
    """
    ### ホログラムを表示する関数
    """
    ...


def main():
    start = time.time()
    constants = QuantumConstants()
    cl_constants = ClassicalConstants()

    ##==  TEST
    circuit = define_regs(qconsts=constants)
    circuit = init_superposition_state(
        circuit=circuit, consts=constants, test=constants.TEST
    )
    ##==

    # points = create_single_point(constants=cl_constants)  # regs?
    # hologram_q = generate_hologram_q(points=points, qconstants=constants)

    # end = time.time()
    # print(print("Cal time:{} sec".format(end - start)))

    # show(hologram_q)


if __name__ == "__main__":
    main()

# 残り TODO
# 1. 測定結果の確認
#       古典計算、手計算の値と合うか確認する
# 2. MULのextentionを作成しSQRにする
# 3. デコレーターの実装. 関数の前後にログを出力する関数を作る
