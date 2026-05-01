from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cv2


def create_single_point(x, y, z) -> np.ndarray:
    """
    デバッグ用 1点の点群を作成する関数

    - X*Yの中心に物体点 (1点) がある想定

    :param constants: 定数クラスのオブジェクト
    :type constants: ClassicalConstants
    :return: デバッグ用の物体点 (1点)
    :rtype: np.ndarray
    """
    x0 = x / 2
    y0 = y / 2
    z0 = z  # 物体点までの距離

    return np.array([[x0, y0, z0]], dtype=float)


def load_image(path: str) -> np.ndarray:
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 輝度値のみ
    if img_gray is not None:
        return img_gray / 256
    raise IOError  # 0-index


def create_image():
    xx = np.arange(-256, 256)  # 画像の中心を(0,0)とする
    yy = np.arange(-256, 256)
    np.meshgrid(xx, yy, indexing="ij")


def a1(grey_image: np.ndarray) -> np.ndarray:
    # intense...伝播元の画像のI_aの各ピクセルの光波の強度分布(画素値)
    # jpeg ガンマ補正
    gamma = 2.2
    intensity = np.power(grey_image, gamma)
    return np.sqrt(intensity)


def p1(phase: np.ndarray) -> np.ndarray:
    # phase...伝播元の画像のI_aの各ピクセルの光波の位相分布(輝度値)
    # 256で割って正規化 -> load_image()で完了している
    return phase


def u1(image: np.ndarray):
    # (2.29)式の実装
    _a1 = a1(image)
    _p1 = p1(image)
    print(f"{np.max(_p1)=}, {np.min(_p1)=}")
    phase = 1j + 2 * np.pi * _p1
    cos_part = _a1 * np.cos(phase)
    sin_part = 1j * _a1 * np.sin(phase)
    return cos_part + sin_part


def h(
    z: float, λ: float, r: np.ndarray, W: int, H: int, pp: float
) -> np.ndarray:
    # 角スペクトル法 (2.23)式の実装
    # z ... z21
    # r...r21

    fx = np.fft.fftfreq(W, d=pp)
    fy = np.fft.fftfreq(H, d=pp)
    Fx, Fy = np.meshgrid(fx, fy)

    cond = (Fx**2 + Fy**2) <= (1 / λ**2)
    func = z * np.sqrt((1 / λ**2) - Fx**2 - Fy**2)
    p = np.where(cond, func, 0)
    return np.exp(1j * 2 * np.pi * p)


def show_twin(
    hologram: np.ndarray, recon: np.ndarray, pp: float, λ: float, d: float
) -> None:
    plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fig.canvas.manager.set_window_title(f"pp={pp}, λ={λ}, d={d}")  # type: ignore
    ax[0].imshow(hologram, cmap="gray")
    ax[0].set_title(f"Hologram")
    ax[0].axis("off")

    intensity = np.abs(recon) ** 2
    ax[1].imshow(intensity, cmap="gray")
    ax[1].set_title("Reconstruction")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    pp_arr = [1.5e-6, 2.2e-6, 3.45e-6, 8.0e-6, 20.0e-6]  # e-6
    λ_arr = [441e-9, 488e-9, 520e-9, 532e-9, 632e-9, 650e-9]  # e-9

    # pp = 8.0e-6
    # λ = 632e-9
    d = 90e-3  # 自動計算

    # u1の作成
    image = load_image("./sample/wavefront/images/orange.jpg")
    W, H = image.shape
    _u1 = u1(image)

    for pp in pp_arr:
        for λ in λ_arr:
            pp = pp
            # 座標作成
            x = np.arange(-W, W) * pp
            y = (
                np.arange(-H, H) * pp
            )  # ゼロパディングに備えて2倍の座標幅を用意する
            u1_x, u1_y = np.meshgrid(x, y)
            u2_x, u2_y = np.meshgrid(x, y)

            # ① u1のゼロパディング
            _u1_pad = np.pad(_u1, W // 2)

            # ② ①をフーリエ変換する
            fa = np.fft.fft2(_u1_pad)

            # ③ 伝達関数Ηを計算する
            dx = u2_x - u1_x * pp
            dy = u2_y - u1_y * pp
            r = np.sqrt(dx**2 + dy**2 + d**2)
            fb = h(z=d, λ=λ, r=r, W=W * 2, H=H * 2, pp=pp)

            # ④ ②と③を複素乗算する / ⑤ ④の結果を逆フーリエ変換する
            out = np.fft.ifft2((fa * fb))

            show_twin(np.angle(out), out, pp, λ, d)


if __name__ == "__main__":
    main()
