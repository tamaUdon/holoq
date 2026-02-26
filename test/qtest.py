import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.visualization import plot_histogram

# pythonリストでの表現
ket0 = np.array([[1], [0]])
ket1 = np.array([[0], [1]])
ket2 = np.array([[0], [2]])

ketmul = np.matmul(ket0, ket1)
print(f"ketmul={ketmul}")


M1 = np.array([[1, 1], [0, 0]])
M2 = np.array([[1, 0], [0, 1]])
M = M1 / 2 + M2 / 2
print(f"M={M}")

# 状態ベクトルの定義
u = Statevector([1 / sqrt(2), 1 / sqrt(2)])
v = Statevector([(1 + 2.0j) / 3, -2 / 3])
w = Statevector([1 / 3, 2 / 3])
print(f"u={u.draw('text')}")
print(f"u={u.draw('latex_source')}")

# 測定 標準ベーシスの測定シミュレート measure
outcome, state = v.measure()
print(f"Measured: {outcome}\nPost-measurement state:")
print(state.draw("latex_source"))

# 測定 sample_counts
statistics = v.sample_counts(1000)
plot_histogram(statistics)
plt.show()

# StateVectorの演算 Operator
Y = Operator([[0, -1.0j], [1.0j, 0]])
H = Operator([[1 / sqrt(2), 1 / sqrt(2)], [1 / sqrt(2), -1 / sqrt(2)]])
S = Operator([[1, 0], [0, 1.0j]])
T = Operator([[1, 0], [0, (1 + 1.0j) / sqrt(2)]])

# ユニタリー演算　evolve
v = Statevector([1, 0])

v = v.evolve(H)
v = v.evolve(T)
v = v.evolve(H)
v = v.evolve(S)
v = v.evolve(Y)

print(f"v={v.draw('latex_source')}")

# 量子回路のプレビュー
circuit = QuantumCircuit(1)
circuit.h(0)
circuit.t(0)
circuit.h(0)
circuit.s(0)
circuit.y(0)

print(circuit.draw(output="mpl"))
plt.show()
