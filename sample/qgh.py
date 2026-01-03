import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from pointcloud import Constants, create_rectangle_points
from reconst_hologram import show_twin
from monopolar import monopolar
from scipy.linalg import norm

qip_dir = Path(__file__).resolve().parents[1] / "quantum-image-processing"
if str(qip_dir) not in sys.path:
    sys.path.insert(0, str(qip_dir))

import circuit  # noqa: E402
import encoder  # noqa: E402

# 1量子ビット演算
ψ = np.zeros([2, 2, 2, 2])
ψ[0, 0, 0, 0] = ψ[1, 1, 1, 1] = 1 / np.sqrt(2)

# 2量子ビット演算
CNOT_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
CNOT_tensor = np.reshape(CNOT_matrix, (2, 2, 2, 2))

# アダマールゲート
H_matrix = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])


class QRegister:
    # TODO - 数式を見ながらQRegistarを再実装する
    # Qiskitを使う
    # エンコードの参考 - https://github.com/ica574/quantum-image-processing
    # Qiskit で NEQR と QFT を使う実装に変更する
    #  - https://github.com/Qiskit/textbook/blob/main/notebooks/ch-applications/image-processing-frqi-neqr.ipynb
    #  - https://qiita.com/tatsunidas/items/ade03830bff751bd7f00

    def __init__(self, n) -> None:
        self.n = n
        self.ψ = np.zeros((2,) * n)  # 初期化
        self.ψ[(0,) * n] = 1  # ψ[0,0,...,0]を1に置き換える


def H(i, reg: QRegister) -> QRegister:
    # アダマールゲートを作用させる関数
    # reg.ψ = np.tensordot(H_matrix, reg.ψ, (1, i))
    # reg.ψ = np.moveaxis(reg.ψ, 0, i)
    reg = np.tensordot(H_matrix, reg, (1, i))
    reg = np.moveaxis(reg, 0, i)
    return reg


def CNOT(control: int, target: int, reg: QRegister) -> QRegister:
    # def H の一般化, 2量子ビット対応
    reg.ψ = np.tensordot(CNOT_tensor, reg.ψ, ((2, 3), (control, target)))
    reg.ψ = np.moveaxis(reg.ψ, (0, 1), (control, target))
    return reg


def generate_GHZ(reg: QRegister) -> QRegister:
    reg = H(0, reg)
    for i in range(reg.n - 1):
        reg = CNOT(i, i + 1, reg)
    return reg


def measure(i, reg: QRegister):
    projectors = [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])]

    def project(i, j, reg: QRegister):
        projected = np.tensordot(projectors[j], reg.ψ, (1, i))
        return np.moveaxis(projected, 0, i)

    projected = project(i, 0, reg)
    norm_projected = norm(projected.flatten())
    if np.random.random() < norm_projected**2:
        reg.ψ = projected / norm_projected
        return 0
    else:
        projected = project(i, 1, reg)
        reg.ψ = projected / norm(projected)
        return 1


def cgh_to_qgh(points: np.ndarray, constants: Constants) -> np.ndarray:
    # 1/√N ∑(j=0, N−1) |aj⟩|Pj⟩ ⊗ |xj⟩|yj⟩ ... Eq(3)
    # initialize all qubits of Eq. (3) to zero.
    # |aj⟩ , |Pj⟩ , |xj⟩ and |yj⟩ denote the quantum registers (collection of qubits) for the point-cloud data.

    scale = 1 << constants.bits_w
    target_bit = constants.bits_w - 1

    # initialize all qubits
    x = np.arange(constants.X, dtype=np.float64)
    y = np.arange(constants.Y, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    holoq = np.zeros((constants.Y, constants.X), dtype=np.float64)  # a?
    P = np.zeros((constants.Y, constants.X), dtype=np.float64)  # P?

    for xj, yj, zj in tqdm.tqdm(points):
        # xj, yjにアダマールゲートをかけて、重畳状態(superposition)にする?
        # Eq. (3).の重畳状態を作る時にcontrolled-NOT ゲートを通す
        # 古典情報 aj , ρj , xh and yh が量子状態になる -> xj, yjを使って計算すれば量子計算になる...ということ?
        # 参考 - https://www.kattemolle.com/other/QCinPY.html

        xj = generate_GHZ(xj)
        yj = generate_GHZ(yj)

        # 以下monopolar
        dx = xx.astype(np.float64) - xj
        dy = yy.astype(np.float64) - yj
        w1 = np.round(dx * dx + dy * dy + zj * zj).astype(np.int64)
        w1 = w1 & ((1 << constants.bits_w) - 1)
        theta = (constants.pp * constants.pp) / (2.0 * constants.λ * zj)
        w2 = int(round(theta * scale))
        theta = w1 * w2
        t = (theta >> target_bit) & 1

        holoq += t.astype(np.float64)
    return holoq


def main():
    # TODO - 初期化~測定を実装

    start = time.time()

    constants = Constants()
    points = create_rectangle_points(constants)
    # cbits = monopolar(points, constants)

    # Preparing Qbits
    # 1/√N ∑(j=0, N−1) |aj⟩|Pj⟩ ⊗ |xj⟩|yj⟩
    # initialize all qubits of Eq. (3) to zero.
    # |aj⟩ , |Pj⟩ , |xj⟩ and |yj⟩ denote the quantum registers (collection of qubits) for the point-cloud data.

    # Next, the coordinates xj and yj of the point cloud are converted into a quantum superposition state using Hadamard gates.
    # Then, the classical information of aj , ρj , xh and yh is converted to qubits through controlled-NOT gates to create the superposition state described in Eq. (3).

    # Computating QGH
    for i in range(100):
        holoq = cgh_to_qgh(points, constants)
        print(holoq.ψ.flatten())
        print(measure(i, holoq), end="")
        print(
            measure(i, holoq), end=" "
        )  # TODO - これを重ね合わせてmonopolarと同じ像を作る

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))
    print("CGH Calculation completed!")

    print("Preparing for display...")
    # show_twin(qhologram, recon) # TODO - 重ね合わせた像を表示


if __name__ == "__main__":
    main()


### TODO
# 量子情報のエンコード
# 1. 点群の古典的bit -> monopolar -> ok
# 2. 点群に対する量子ビットの準備 (基底エンコーディング) -> now
# 3. QGHの計算 (QFTベースの回路) -> ok
# 4. ホログラムピクセルの測定 (Qbitの測定はポップカウントと同等)
