import math
import time
import tqdm
import open3d.data
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pointcloud import Constants, create_rectangle_points
from reconst_hologram import show_twin
from monopolar import monopolar
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.visualization import plot_histogram
from numpy import sqrt


def init_qbits(cbits: np.ndarray, constants: Constants) -> np.ndarray:
    # 1/√N ∑(j=0, N−1) |aj⟩|Pj⟩ ⊗ |xj⟩|yj⟩
    # initialize all qubits of Eq. (3) to zero.
    # |aj⟩ , |Pj⟩ , |xj⟩ and |yj⟩ denote the quantum registers (collection of qubits) for the point-cloud data.

    # initialize all qubits
    x = np.arange(constants.X, dtype=np.float64)
    y = np.arange(constants.Y, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    hologram = np.zeros((constants.Y, constants.X), dtype=np.float64)  # a?
    P = np.zeros((constants.Y, constants.X), dtype=np.float64)  # P?

    for xj, yj, zj in tqdm.tqdm(cbits):
        # xj, yjにアダマールゲートをかけて、重畳状態(superposition)にする?
        # Eq. (3).の重畳状態を作る時にcontrolled-NOT ゲートを通す
        # 古典情報 aj , ρj , xh and yh が量子状態になる -> xj, yjを使って計算すれば量子計算になる...ということ?
        # 参考 - https://www.kattemolle.com/other/QCinPY.html

        # 以下monopolar
        dx = xx.astype(np.float64) - xj
        dy = yy.astype(np.float64) - yj
        w1 = np.round(dx * dx + dy * dy + zj * zj).astype(np.int64)
        w1 = w1 & ((1 << constants.bits_w) - 1)
        theta = (constants.pp * constants.pp) / (2.0 * constants.λ * zj)
        w2 = int(round(theta * scale))
        theta = w1 * w2
        t = (theta >> target_bit) & 1

        hologram += t.astype(np.float64)
    return hologram


def apply_hadamar(iqs: np.ndarray):
    ...
    # Next, the coordinates xj and yj of the point cloud are converted into a quantum superposition state using Hadamard gates.
    # Then, the classical information of aj , ρj , xh and yh is converted to qubits through controlled-NOT gates to create the superposition state described in Eq. (3).


def qgh(qbits: np.ndarray): ...


def measure(qhologram: np.ndarray): ...


def main():
    start = time.time()

    constants = Constants()
    points = create_rectangle_points(constants)
    cbits = monopolar(points, constants)

    # Preparing Qbits
    iqs = init_qbits(cbits, constants)
    qbits = apply_hadamar(iqs)

    # Computating QGH
    qhologram = qgh(qbits)
    recon = measure(qhologram)

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))
    print("CGH Calculation completed!")

    print("Preparing for display...")
    show_twin(qhologram, recon)


if __name__ == "__main__":
    main()


### TODO
# 量子情報のエンコード
# 1. 点群の古典的bit -> monopolar -> ok
# 2. 点群に対する量子ビットの準備 (基底エンコーディング) -> now
# 3. QGHの計算 (QFTベースの回路)
# 4. ホログラムピクセルの測定 (Qbitの測定はポップカウントと同等) <--- 担当と思われる箇所
