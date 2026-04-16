import time

import numpy as np
import tqdm
from constants import ClassicalConstants
from pointcloud import create_rectangle_points, show


def monopolar_numpy(points: np.ndarray, constants: ClassicalConstants):
    # numpy実装版 - 512*512画素で7sec
    x = np.arange(constants.X, dtype=np.float64) * constants.pp
    y = np.arange(constants.Y, dtype=np.float64) * constants.pp
    xx, yy = np.meshgrid(x, y)
    hologram = np.zeros((constants.Y, constants.X), dtype=np.float64)

    for xj, yj, zj in tqdm.tqdm(points):
        hx = xx - xj * constants.pp
        hy = yy - yj * constants.pp
        rho = constants.k / zj
        phase = rho * (hx * hx + hy * hy + zj * zj)
        hologram += np.where(np.cos(phase) >= 0.0, 1.0, -1.0) # cosあり

    return hologram


def main():
    start = time.time()

    constants = ClassicalConstants()
    points = create_rectangle_points(constants)
    hologram = monopolar_numpy(points, constants)

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))
    print("CGH Calculation completed!")

    print("Preparing for display...")
    show(hologram, constants.X, constants.Y, binary=False, save=False)


if __name__ == "__main__":
    main()
