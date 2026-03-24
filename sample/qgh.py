# 点群法で量子コンピュータ生成ホログラムを作る

import time
import math
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from constants import QuantumConstants, ClassicalConstants
from pointcloud import create_single_point
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram, plot_distribution
from qiskit.circuit import QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import (
    DraperQFTAdder,
    RGQFTMultiplier,
)


def define_gates(qconsts: QuantumConstants) -> tuple:
    """
    ゲートの定義
    """
    adder = DraperQFTAdder(num_state_qubits=qconsts.W, kind="half")
    adder_sum = DraperQFTAdder(num_state_qubits=qconsts.mul_w, kind="half")
    mul = RGQFTMultiplier(
        num_state_qubits=qconsts.sq_w, num_result_qubits=qconsts.res_w
    )
    sqr = RGQFTMultiplier(
        num_state_qubits=qconsts.add_w,
        num_result_qubits=qconsts.mul_w,
        name="SQR_RGQFTMultiplier",
    )
    return adder, adder_sum, mul, sqr


def define_regs(qconsts: QuantumConstants) -> QuantumCircuit:
    """
    レジスタの定義
    """

    xj_reg = QuantumRegister(qconsts.W, "xj")
    xh_reg = QuantumRegister(qconsts.W, "xh")
    xhj_reg = AncillaRegister(qconsts.add_w, "xhj")
    xhj_b_reg = AncillaRegister(qconsts.add_w, "xhj_b")
    xhj_sq_reg = AncillaRegister(qconsts.mul_w, "xhj_sq_reg")
    yj_reg = QuantumRegister(qconsts.W, "yj")
    yh_reg = QuantumRegister(qconsts.W, "yh")
    yhj_reg = AncillaRegister(qconsts.add_w, "yhj")
    yhj_b_reg = AncillaRegister(qconsts.add_w, "yhj_b")
    yhj_sq_reg = AncillaRegister(qconsts.sq_w, "yhj_sq_reg")
    rho_reg = QuantumRegister(qconsts.sq_w, "rho")
    result = AncillaRegister(qconsts.res_w, "result")
    cl_result = ClassicalRegister(qconsts.res_w, "cl_result")

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


def load_integer(circuit: QuantumCircuit, reg: QuantumRegister, value: int):
    assert value < 2 ** len(reg)
    for i in range(len(reg)):
        if (value >> i) & 1:
            circuit.x(reg[i])


def init_superposition_state(
    circuit: QuantumCircuit,
    xh: int,
    yh: int,
    points: list[tuple[int, int]],
    rho_values: list[int],
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

    """
    ## WARNING - 固定値を用いたテスト実装版
    -  a_j = [1,2,3,0] # -> aは一旦無視して計算する
    - xj_yj = [(0, 0), (1, 0), (0, 1), (1, 1)]
    - rho_j = [0.5, 0.25, 0.5, 0]
    -      -> [10, 01, 10, 00] として扱う
    """

    # 1. 重ね合わせを作る
    assert len(xj_reg) == len(yj_reg)
    for xj, yj in zip(xj_reg, yj_reg):
        circuit.h(xj)
        circuit.h(yj)

    # 2. CNOTを用いてρ_jのビットを反転させる
    controls = [xj_reg, yj_reg]

    for (px, py), rho in zip(points, rho_values):
        ctrl_state = format(px, f"0{len(xj_reg)}b") + format(py, f"0{len(yj_reg)}b")

        for bit_index in range(len(rho_reg)):
            if (rho >> bit_index) & 1:
                circuit.mcx(controls, rho_reg[bit_index], ctrl_state=ctrl_state)

    # circuit.mcx(controls, rho_reg[1], ctrl_state="00")  # |10> 0.5
    # circuit.mcx(controls, rho_reg[0], ctrl_state="10")  # |01> 0.25
    # circuit.mcx(controls, rho_reg[1], ctrl_state="01")  # |10> 0.5

    # 3. xh, yhに値を入れる
    load_integer(circuit, xh_reg, xh)
    load_integer(circuit, yh_reg, yh)

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
    circuit.append(
        sqr, list(yhj_reg) + list(yhj_sub_reg) + list(yhj_sq_reg[:-1])
    )  # [:-1] 次のADDに使う分を空けておく

    # ③ ADD
    circuit.append(adder_sum, list(xhj_sq_reg) + list(yhj_sq_reg))

    # ④ MUL
    circuit.append(mul, list(yhj_sq_reg) + list(rho_reg) + list(result_reg))

    # 測定前にbarrier
    circuit.barrier()

    # MEASURE
    # circuit.measure_all()  # デバッグ用. 全てのビットを確認する
    circuit.measure(result_reg[0], cl_result[0])
    return circuit


def execute(circuit: QuantumCircuit) -> dict:
    """
    ### 回路をシミュレートする関数

    :param circuit: 量子回路のインスタンス
    :param type: QuantumCircuit
    """
    simulator = AerSimulator(
        method="matrix_product_state"
    )  # StateVectorで検証すると動かなかったのでMPSで試した

    transpiled_circuit = transpile(
        circuit,
        simulator,
        coupling_map=None,  # WARNING - 検証用, 実機での実行時は指定必須
        optimization_level=1,
    )

    job = simulator.run(transpiled_circuit, shots=1)
    result = job.result()
    counts = result.get_counts(circuit)  # qubit = anc5[0]をカウント

    return counts


def build_circuit(
    qconsts: QuantumConstants,
    bw: int,
    gates: tuple,
    xh: int,
    yh: int,
    points: list[tuple[int, int]],
    rho_values: list[int],
) -> QuantumCircuit:
    circuit = define_regs(qconsts=qconsts)
    circuit = init_superposition_state(
        circuit=circuit,
        xh=xh,
        yh=yh,
        points=points,
        rho_values=rho_values,
    )
    circuit = define_circuit(circuit=circuit, qgates=gates, test=qconsts.TEST)
    return circuit


def generate_hologram_q(
    qconsts: QuantumConstants,
    bw: int,
    points: list[tuple[int, int]],
    rho_values: list[int],
) -> np.ndarray:
    gates = define_gates(qconsts=qconsts)
    hologram = np.zeros((qconsts.Y, qconsts.X), dtype=int)

    for xh in tqdm.tqdm(range(qconsts.X)):
        for yh in range(qconsts.Y):
            circuit = build_circuit(
                qconsts=qconsts,
                gates=gates,
                bw=bw,
                xh=xh,
                yh=yh,
                points=points,
                rho_values=rho_values,
            )  # 測定ごとに回路を作り直す?
            # print(circuit.draw("text")) # 回路の確認用

            result: dict = execute(circuit=circuit)
            logg(result)
            T = next(iter(result))[:2]  # WARNING - 最初のkeyの0,1番目の文字を取り出す
            hologram[xh, yh] = T
    return hologram


def show(hologram: np.ndarray) -> None:
    """
    ### ホログラムを表示する関数
    """
    print(hologram)
    plt.imshow(hologram, cmap="gray", origin="lower", interpolation="nearest")
    plt.colorbar()
    plt.title("Hologram")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def logg(counts: dict) -> None:
    integer_counts = {}
    for binary_string, count in counts.items():
        print(f"{binary_string=}")  # 1,0のような文字列が入っている
        integer_value = int(binary_string[0], 2)
        integer_counts[integer_value] = count
    print(f"Measurement counts (binary strings): {counts}")
    print(f"Result register counts (integers): {integer_counts}")

    ## TEST - 確率確認用
    # plot_distribution(counts)
    # plt.show()


def main():
    print(print("Start calculation..."))
    start = time.time()

    q_constants = QuantumConstants(N=4, X=10)
    points = [(0, 0), (1, 0), (0, 1), (1, 1)]  # 4点
    rho_values = [0b10, 0b01, 0b10, 0b00]  # 0.0 -> 00, 0.25->01, 0.5->10, 0.75-> 11

    hologram = generate_hologram_q(
        qconsts=q_constants, bw=q_constants.W, points=points, rho_values=rho_values
    )

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))

    show(hologram)


if __name__ == "__main__":
    main()

# 残り TODO
# 1. 測定結果の確認
#       古典計算、手計算の値と合うか確認する
# 2. MULのextentionを作成しSQRにする
# 3. デコレーターの実装. 関数の前後にログを出力する関数を作る
# 4. 点群を読み込む関数を作成する
# 5. ホログラムを表示する関数を作成する

# 改善 TOOD
# 1. BitWidthの初期化 Nを渡すところ もっといい実装はないか?
