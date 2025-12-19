import dataclasses
import math
import time

import matplotlib.pyplot as plt
import numpy as np

from pointcloud import Constants, create_rectangle_points, generate_hologram


def fresnel_propagation(
    field: np.ndarray, constants: Constants, z: float
) -> np.ndarray:
    fx = np.fft.fftfreq(constants.X, d=constants.pp)
    fy = np.fft.fftfreq(constants.Y, d=constants.pp)
    fx, fy = np.meshgrid(fx, fy)
    h = np.exp(1j * constants.k * z) * np.exp(
        -1j * math.pi * constants.λ * z * (fx * fx + fy * fy)
    )
    spectrum = np.fft.fft2(field)
    return np.fft.ifft2(spectrum * h)


def show(hologram: np.ndarray, recon: np.ndarray) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(hologram, cmap="gray")
    ax[0].set_title("Hologram")
    ax[0].axis("off")

    intensity = np.abs(recon) ** 2
    ax[1].imshow(intensity, cmap="gray")
    ax[1].set_title("Reconstruction")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    start = time.time()
    constants = Constants()
    points = create_rectangle_points(constants)

    hologram = generate_hologram(points, constants)
    recon = fresnel_propagation(hologram.astype(np.complex128), constants, constants.d)

    end = time.time()
    print(f"Cal time: {end - start:.3f} sec")
    show(hologram, recon)


if __name__ == "__main__":
    main()
