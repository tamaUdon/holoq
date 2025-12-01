import math
import time
import tqdm
import open3d.data
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import dataclasses


@dataclasses.dataclass
class Constants:
    """
    Constants の Docstring

    :param DEBUG: デバッグON/OFF
    :type DEBUG: Bool
    :param X,Y: 画素数
    :type X,Y: int
    :param λ: 波長
    :type λ: int (nm) # TODO - [int]にする
    :param k: 波数 (2pi/λ)
    :type k: int
    :param pp: 画素ピッチ
    :type pp: int
    :param d: ホログラムと物体間の距離 # TODO - 空間分解能を考慮して実装する
    :type d: int

    :return: bipolarホログラムの計算結果
    :rtype: np.ndarray
    """

    DEBUG = True
    X = 2000  # 画素X方向
    Y = X
    λ = 500  # 波長
    k = 2 * math.pi / λ
    pp = 10e-6  # μm
    d = 10e6  # μm


def create_pinhole_camera_parameters(
    pcd: open3d.geometry.PointCloud,
) -> open3d.camera.PinholeCameraIntrinsic:
    # 参考 - https://zenn.dev/fastriver/articles/open3d-camera-pinhole#%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E3%82%92%E5%8F%96%E5%BE%97%E3%81%99%E3%82%8B
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    view_control = vis.get_view_control()
    pinhole_parameters = view_control.convert_to_pinhole_camera_parameters()

    print(pinhole_parameters.intrinsic.intrinsic_matrix)
    print(pinhole_parameters.extrinsic)

    return pinhole_parameters.intrinsic


def create_point_cloud(coordinate: np.ndarray) -> open3d.geometry.PointCloud:
    data = np.array(coordinate)
    pcd = o3d.geometry.create_from_depth_image()
    pcd.points = o3d.utility.Vector3dVector(data)

    pinhole_parameters = create_pinhole_camera_parameters(pcd)
    pcd = o3d.geometry.create_from_depth_image(pcd, pinhole_parameters)
    return pcd


def load_bunny_pointcloud() -> open3d.geometry.PointCloud:
    bunny_path = open3d.data.BunnyMesh().path
    point_cloud = o3d.io.read_point_cloud(bunny_path)
    print("Loading data completed!")
    return point_cloud


def downsampling(
    point_cloud: open3d.geometry.PointCloud, every_k_points: int = 10
) -> open3d.geometry.PointCloud:
    points = point_cloud.uniform_down_sample(every_k_points=every_k_points)
    print("Downsampling completed!")
    return points


def calculate_holography(data: np.ndarray, constants: Constants) -> np.ndarray:
    I_holography = np.zeros((constants.Y, constants.X))

    print("Calculating CGH...")
    for y_i in tqdm.tqdm(range(constants.Y)):
        for x_i in range(constants.X):
            for dt in data:
                x_j = dt[0]
                y_j = dt[1]
                z_j = dt[2]

                x_p = ((x_i) * constants.pp - x_j * constants.pp) ** 2
                y_p = ((y_i) * constants.pp - y_j * constants.pp) ** 2
                z_p = z_j**2

                r = math.sqrt((x_p + y_p + z_p))
                I_tmp = (1 / r) * math.cos(constants.k * r)
                I_holography[y_i, x_i] = I_tmp
    print("CGH Calculation completed!")
    return I_holography


def show_hologram(I_holography: np.ndarray) -> None:
    print("Preparing for display...")

    fig, ax = plt.subplots()
    CS = ax.contourf(range(Constants.X), range(Constants.Y), I_holography)
    fig.colorbar(CS)
    fig.set_label("holography")
    plt.legend()
    plt.show()


def main():
    start = time.time()
    print("Preparing for CGH...")

    constants = Constants()
    point_cloud = np.array([[0, 0, 0]])

    if constants.DEBUG:
        coord = np.array(
            [[constants.X / 2, constants.Y / 2, 10e-3]]  #
        )  # DEBUG用の1点
        point_cloud = create_point_cloud(coord)
    else:
        point_cloud = load_bunny_pointcloud()
        point_cloud = downsampling(point_cloud, every_k_points=1000)

    points = np.asarray(point_cloud.points)
    holography = calculate_holography(points, constants)

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))

    show_hologram(holography)


if __name__ == "__main__":
    main()

# TODO
# 1. コマンドライン引数を受け取れるようにする
# 2. Constantsに引数データを入れる
# 3. 複数波長を受け取れるようにλを[]にする
