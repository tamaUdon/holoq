import numpy as np
import tqdm
import time
import matplotlib.pyplot as plt
from constants import Constants
from pointcloud import create_rectangle_points, show


def generate_hologram(points: np.ndarray, constants: Constants) -> np.ndarray:
    x = np.arange(constants.X, dtype=np.float64) * constants.pp
    y = np.arange(constants.Y, dtype=np.float64) * constants.pp
    xx, yy = np.meshgrid(x, y)
    hologram = np.zeros((constants.Y, constants.X), dtype=np.float64)

    for xj, yj, zj in tqdm.tqdm(points):
        dx = xx - xj * constants.pp
        dy = yy - yj * constants.pp
        r = np.sqrt(dx * dx + dy * dy + zj * zj)
        hologram += np.cos(constants.k * r) / r
    return hologram


def bipolar(points: np.ndarray, constants: Constants):
    x = np.arange(constants.X, dtype=np.float64) * constants.pp
    y = np.arange(constants.Y, dtype=np.float64) * constants.pp
    xx, yy = np.meshgrid(x, y)
    hologram = np.zeros((constants.Y, constants.X), dtype=np.float64)

    for xj, yj, zj in tqdm.tqdm(points):
        hx = xx - xj * constants.pp
        hy = yy - yj * constants.pp
        rho = constants.k / zj
        phase = rho * (hx * hx + hy * hy + zj * zj)
        hologram += np.where(
            np.cos(phase) >= 0.0, 1.0, -1.0
        )  # TODO - この行はcosを使わない
    return hologram


def main():
    start = time.time()
    constants = Constants()

    points = create_rectangle_points(constants)  # 四角形 # TODO - 分岐
    hologram = generate_hologram(points, constants)
    print("CGH Calculation completed!")

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))

    print("Preparing for display...")
    show(hologram)


if __name__ == "__main__":
    main()
