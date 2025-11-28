import math
import time
import tqdm
import open3d.data
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pointcloud import Constants


def load_classical_pointcloud() -> open3d.geometry.PointCloud:
    bunny_path = open3d.data.BunnyMesh().path
    point_cloud = o3d.io.read_point_cloud(bunny_path)
    print("Loading data completed!")
    return point_cloud


def calculate_bipolar_holography(data: np.ndarray, constants: Constants) -> np.ndarray:
    """
    calculate_bipolar_holography の Docstring

    :param data: 点群データ
    :type data: np.ndarray
    :param constants: 定数データクラス
    :type constants: Constants
    :return: bipolarホログラムの計算結果
    :rtype: np.ndarray
    """

    I_holography = np.zeros((constants.Y, constants.X))

    print("Calculating CGH...")
    for y_i in tqdm.tqdm(range(constants.Y)):
        for x_i in range(constants.X):
            for d in data:
                x_j = d[0]
                y_j = d[1]
                z_j = d[2]

                x_p = ((constants.PXL * x_i) - x_j) ** 2
                y_p = ((constants.PXL * y_i) - y_j) ** 2
                z_p = z_j**2

                r = math.sqrt((x_p + y_p + z_p))  # TODO - Bipolarに変更する
                I_tmp = (1 / r) * math.cos(constants.k * r)
                I_holography[y_i, x_i] = I_tmp
    print("CGH Calculation completed!")
    return I_holography


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
