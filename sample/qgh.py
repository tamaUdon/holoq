# 参考 - https://github.com/Qiskit/textbook/blob/main/notebooks/ch-applications/image-processing-frqi-neqr.ipynb

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.visualization import plot_histogram

import qiskit as qk
from math import pi
import numpy as np


# Initialize the quantum circuit for the image
# Pixel position
# 量子回路を初期化
idx = QuantumRegister(2, "idx")

# グレースケールの場合
# grayscale pixel intensity value
intensity = QuantumRegister(8, "intensity")

# 古典レジスタ
# classical register
cr = ClassicalRegister(10, "cr")  # 8bitグレースケール + 量子2bit?

# 量子回路を作成
# create the quantum circuit for the image
qc_image = QuantumCircuit(intensity, idx, cr)

# 量子ビットの個数を設定
num_qubits = qc_image.num_qubits

qc_image.draw()

###########

# Initialize the quantum circuit

# Optional: Add Identity gates to the intensity values
# オプション: 識別ゲートを追加
for idx in range(intensity.size):
    qc_image.id(idx)

# Add Hadamard gates to the pixel positions
qc_image.h(8)
qc_image.h(9)

# Separate with barrier so it is easy to read later.
qc_image.barrier()
qc_image.draw()

###########

# Encode the first pixel, since its value is 0, we will apply ID gates here:
# 最初のピクセルをエンコード、値が０なのでIDゲートを適用するのみ
for idx in range(num_qubits):
    qc_image.id(idx)

qc_image.barrier()
qc_image.draw()


###########


# Encode the second pixel whose value is (01100100):
# 2番目のピクセルをエンコード (01 = 01100100 (Grayscale = 100))
value01 = "01100100"

# Add the NOT gate to set the position at 01:
# NOTゲートを適用 -> 01ポジションへ
qc_image.x(qc_image.num_qubits - 1)

# We'll reverse order the value so it is in the same order when measured.
for idx, px_value in enumerate(value01[::-1]):
    if px_value == "1":
        qc_image.ccx(num_qubits - 1, num_qubits - 2, idx)

# Reset the NOT gate
# NOT gateをリセット
qc_image.x(num_qubits - 1)

qc_image.barrier()
qc_image.draw()

###########


# Encode the third pixel whose value is (11001000):
# 3番目のピクセルをエンコード
value10 = "11001000"

# Add the 0CNOT gates, where 0 is on X pixel:
qc_image.x(num_qubits - 2)
for idx, px_value in enumerate(value10[::-1]):
    if px_value == "1":
        qc_image.ccx(num_qubits - 1, num_qubits - 2, idx)
qc_image.x(num_qubits - 2)


qc_image.barrier()
qc_image.draw()


###########


# Encode the third pixel whose value is (11111111):
# 3番目のピクセルをエンコーディング
value11 = "11111111"

# Add the CCNOT gates:
for idx, px_value in enumerate(value11):
    if px_value == "1":
        qc_image.ccx(num_qubits - 1, num_qubits - 2, idx)

qc_image.barrier()
qc_image.measure(range(10), range(10))
qc_image.draw()


###########


print("Circuit dimensions")
print("Circuit depth: ", qc_image.decompose().depth())
print("Circuit size: ", qc_image.decompose().size())

qc_image.decompose().count_ops()


###########

aer_sim = AerSimulator()

try:
    t_qc_image = transpile(qc_image, backend=aer_sim, optimization_level=1)
    job_neqr = aer_sim.run(t_qc_image, shots=8192)
except Exception as e:
    print("[warn] transpile failed, running without transpile:", repr(e))
    job_neqr = aer_sim.run(qc_image, shots=8192)

result_neqr = job_neqr.result()

# circuitが1本の場合
# 複数の場合は get_counts(0) で index 指定する
counts_neqr = result_neqr.get_counts()

print("Encoded: 00 = 0")
print("Encoded: 01 = 01100100")
print("Encoded: 10 = 11001000")
print("Encoded: 11 = 1")

print(counts_neqr)
plot_histogram(counts_neqr)
