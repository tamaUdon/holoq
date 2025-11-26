import math
import time
import tqdm
import open3d.data
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def load_classical_pointcloud(): ...


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
