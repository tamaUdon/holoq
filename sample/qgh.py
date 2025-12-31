import math
import time
import tqdm
import open3d.data
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pointcloud import Constants
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.visualization import plot_histogram
from numpy import sqrt


def load_classical_pointcloud() -> open3d.geometry.PointCloud:
    bunny_path = open3d.data.BunnyMesh().path
    point_cloud = o3d.io.read_point_cloud(bunny_path)
    print("Loading data completed!")
    return point_cloud


def encode_object_to_qbits(data: np.ndarray, constants: Constants) -> np.ndarray:
    """
    calculate_bipolar_holography の Docstring

    :param data: 点群データ
    :type data: np.ndarray
    :param constants: 定数データクラス
    :type constants: Constants
    :return: bipolarホログラムの計算結果
    :rtype: np.ndarray
    """

    position_states = np.zeros((constants.Y, constants.X))

    print("Calculating QGH...")
    for y_i in tqdm.tqdm(range(constants.Y)):
        for x_i in range(constants.X):
            for d in data:
                x_j = d[0]
                y_j = d[1]
                z_j = d[2]
                rho_j = constants.k * z_j

                u = Statevector(
                    [
                        x_j,
                        y_j,
                        rho_j,
                    ]
                )
                position_states.append(u)

    # TODO - position_states -> 重ね合わせる

    print("QGH Calculation completed!")
    return position_states


def preprocess_qbit_from_pointcloud(): ...


def calculate_qgh(): ...


def measure_qbit(): ...


### TODO
# 量子情報のエンコード
# 1. 点群の古典的bit -> ok
# 2. 点群に対する量子ビットの準備 (基底エンコーディング) -> 進行中
# 3. QGHの計算 (QFTベースの回路)
# --- ここまでは論文で理論化済み --- #
# 4. ホログラムピクセルの測定 (Qbitの測定はポップカウントと同等) <--- 担当と思われる箇所
