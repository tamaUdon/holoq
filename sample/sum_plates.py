from sample.pointcloud import (
    calculate_zoneplate,
    create_single_point,
    show_graph,
    Constants,
)
import time
import numpy as np
import matplotlib.pyplot as plt

points_n = 2


def sumup_plates(plates: list):
    pass


def main():
    start = time.time()
    print("Preparing for CGH...")

    constants = Constants()
    points = np.array([[0, 0, 0]])
    hologram = np.array()

    points = create_single_point(constants)

    for n in range(points_n):
        plate = calculate_zoneplate(points, constants)
        hologram.additive(plate)  # 畳み込み

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))
    show_graph(hologram)


if __name__ == "__main__":
    main()

# TODO
# 0. ホログラム入門を確認 -> p29, 2.2.4. 複数点のCGHから
# 1. 畳み込み
# 2. パディング
# 3. デバッグ（正しく足し合わされているか）
# 4. 1/r を 2.17式 p28 に変更する
