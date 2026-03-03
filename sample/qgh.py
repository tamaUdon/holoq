# 点群法で量子コンピュータ生成ホログラムを作る

import time
import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
from constants import Constants
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import DraperQFTAdder, MultiplierGate, QFTGate, QFT, HGate


# # 固定値 N=4
# N = 4  # 点群の物体点数3+1つダミーとする
# # a = [1, 2, 3, 0]
# ρ = [0.5, 0.25, 0.5, 0]  # 最後ρ=0なので位相の寄与なし=ダミー、という意味?
# xj_yj = [(0, 0), (1, 0), (0, 1), (1, 1)]
# xh_yh = [(0, 0), (1, 0), (0, 1), (1, 1)]

# 固定値 N=2
N = 2  # 点群の物体点数1+1つダミーとする
# a = [1, 2, 3, 0]
ρ = np.array([(0.5), (0.0)])  # 最後ρ=0なので位相の寄与なし=ダミー、という意味?
xj_yj = np.array([(0.0, 0.5), (0.5, 1.0)])
xh_yh = np.array([(1.0, 0.8), (0.6, 1.0)])


def init_superposition_state(
    bits_width: int,
) -> QuantumCircuit:
    assert ...  # 2の累乗であることを確認

    xh_reg = QuantumRegister(bits_width, "xh")
    anc1 = QuantumRegister(2, "anc1")  # |0>|0>
    yh_reg = QuantumRegister(bits_width, "yh")
    anc2 = QuantumRegister(2, "anc2")  # |0>|0>
    rho_reg = QuantumRegister(bits_width, "rho")
    anc3 = QuantumRegister(1, "anc3")  # |0>

    qc = QuantumCircuit(
        xh_reg, anc1, yh_reg, anc2, rho_reg, anc3
    )  # |xh>|0>|0>|yh>|0>|0>|ρj>|0>

    xh_offset = qc.find_bit(xh_reg[0]).index
    yh_offset = qc.find_bit(yh_reg[0]).index
    rho_offset = qc.find_bit(rho_reg[0]).index
    state = np.zeros(1 << qc.num_qubits, dtype=complex)

    def _float_to_int(value: float):
        """
        :bit幅に合わせて0.0-1.0を刻む補助関数:
        - 4bitの場合 0.3 -> 3 -> 0011にmapする
        """
        return round(value * ((1 << bits_width) - 1))

    for j in range(N):
        xh, yh = xh_yh[j]  # 古典ビット
        rho = ρ[j]
        basis_idx = (
            (_float_to_int(xh) << xh_offset)
            | (_float_to_int(yh) << yh_offset)
            | (_float_to_int(rho) << rho_offset)
        )
        print(f"{basis_idx=}")
        state[basis_idx] += 1 / math.sqrt(N)
    qc.initialize(Statevector(state))
    return qc


def main():
    constants = Constants()
    qc = init_superposition_state(bits_width=constants.bits_w)
    qc.decompose().draw("mpl")
    plt.show()


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
