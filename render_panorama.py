"""
全景渲染脚本
============================================
根据固定距离和间隔角度，生成所有视角的渲染图像。

用法：
    python render_panorama.py

输出：
    ./output/panorama/ - 所有渲染的PNG图像（含alpha通道）
"""

import numpy as np
import trimesh
import pyrender
import cv2
import os
from typing import Tuple, Optional, List

# ============================================================================
# 配置参数
# ============================================================================

MESH_PATH = "./model.glb"           # 3D网格文件路径
PANORAMA_OUTPUT_DIR = "./output/panorama"  # 全景输出目录

# 渲染参数
RENDER_WIDTH = 256
RENDER_HEIGHT = 256
CAMERA_FOV = 90.0

# 相机参数
CAMERA_DISTANCE = 1.5  # 相对于网格对角线的倍数（固定）
AZIMUTH_INTERVAL = 5   # 方位角间隔（度）
ELEVATION_INTERVAL = 5 # 仰角间隔（度）
ELEVATION_RANGE = (-60, 60)  # 仰角范围（度）
REVERSE_AZIMUTH = False  # 是否反转方位角方向（如果看不到模型则改为True）


def create_axis_mesh(origin: np.ndarray, axis_length: float = 100) -> List[Tuple[pyrender.Mesh, np.ndarray]]:
    """
    创建坐标轴可视化（RGB三条从原点出发的立方体条纹）
    
    参数：
        origin: 坐标轴原点
        axis_length: 坐标轴长度
    
    返回：
        三个 (pyrender.Mesh, pose) 元组列表（X红、Y绿、Z蓝）
    """
    axis_radius = 0.05  # 立方体厚度
    
    # 创建三条轴作为扁平长立方体
    axes_meshes = []
    
    # X轴（红）- 沿 +X 方向，从原点到 +X
    x_box = trimesh.creation.box(
        extents=[axis_length, axis_radius, axis_radius]
    )
    # 立方体默认在原点，平移使其从原点开始
    x_box.apply_translation(origin + np.array([axis_length/2, 0, 0]))
    x_mesh = pyrender.Mesh.from_trimesh(x_box)
    red_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 0.0, 0.0, 1.0],
        roughnessFactor=1.0,
        metallicFactor=0.0
    )
    x_mesh.primitives[0].material = red_material
    axes_meshes.append((x_mesh, np.eye(4)))
    
    # Y轴（绿）- 沿 +Y 方向，从原点到 +Y
    y_box = trimesh.creation.box(
        extents=[axis_radius, axis_length, axis_radius]
    )
    # 立方体默认在原点，平移使其从原点开始
    y_box.apply_translation(origin + np.array([0, axis_length/2, 0]))
    y_mesh = pyrender.Mesh.from_trimesh(y_box)
    green_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.0, 1.0, 0.0, 1.0],
        roughnessFactor=1.0,
        metallicFactor=0.0
    )
    y_mesh.primitives[0].material = green_material
    axes_meshes.append((y_mesh, np.eye(4)))
    
    # Z轴（蓝）- 沿 +Z 方向，从原点到 +Z
    z_box = trimesh.creation.box(
        extents=[axis_radius, axis_radius, axis_length]
    )
    # 立方体默认在原点，平移使其从原点开始
    z_box.apply_translation(origin + np.array([0, 0, axis_length/2]))
    z_mesh = pyrender.Mesh.from_trimesh(z_box)
    blue_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.0, 0.0, 1.0, 1.0],
        roughnessFactor=1.0,
        metallicFactor=0.0
    )
    z_mesh.primitives[0].material = blue_material
    axes_meshes.append((z_mesh, np.eye(4)))
    
    return axes_meshes


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """
    标准 OpenGL/pyrender LookAt 矩阵计算（Y-Up 世界坐标系）。
    """
    if up is None:
        up = np.array([0.0, 1.0, 0.0])

    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    # 标准 OpenGL LookAt 实现
    z_axis = eye - target
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross(np.array([1.0, 0.0, 0.0]), z_axis)
    
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # 构建相机位姿矩阵（Camera Pose）
    # pyrender 需要列向量表示相机坐标系在世界坐标系中的方向
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = x_axis  # 相机的 +X 轴（右方向）
    matrix[:3, 1] = y_axis  # 相机的 +Y 轴（上方向）
    matrix[:3, 2] = z_axis  # 相机的 +Z 轴（后方向）
    matrix[:3, 3] = eye     # 相机位置

    return matrix


class PanoramaRenderer:
    """全景渲染器"""
    
    def __init__(
        self,
        mesh_path: str,
        render_resolution: Tuple[int, int] = (RENDER_WIDTH, RENDER_HEIGHT),
        fov: float = CAMERA_FOV
    ):
        """初始化渲染器"""
        print(f"正在加载网格: {mesh_path}")
        
        self.mesh = trimesh.load(mesh_path, force='mesh')
        self.mesh_center = self.mesh.bounding_box.centroid.copy()
        
        bounds = self.mesh.bounds
        self.mesh_extent = bounds[1] - bounds[0]
        self.mesh_scale = np.linalg.norm(self.mesh_extent)
        
        print(f"  网格中心: {self.mesh_center}")
        print(f"  网格尺寸: {self.mesh_extent}")
        print(f"  对角线长度: {self.mesh_scale:.4f}")
        
        self.render_resolution = render_resolution
        self.fov = fov
        
        # 初始化场景
        self._setup_scene()
    
    def _setup_scene(self):
        """设置渲染场景"""
        self.scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0],
            ambient_light=[0.3, 0.3, 0.3]
        )
        
        # 添加网格（白色材质）
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0
        )
        pyrender_mesh = pyrender.Mesh.from_trimesh(self.mesh, material=material)
        self.mesh_node = self.scene.add(pyrender_mesh)
        
        # 添加坐标轴（以网格中心为原点，极大长度用于无限射线效果）
        axis_meshes = create_axis_mesh(self.mesh_center, axis_length=1e6)
        for mesh, pose in axis_meshes:
            self.scene.add(mesh, pose=pose)
        
        # 添加灯光
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        self.scene.add(light, pose=np.eye(4))
        
        # 创建相机
        self.camera = pyrender.PerspectiveCamera(
            yfov=np.radians(self.fov),
            aspectRatio=self.render_resolution[0] / self.render_resolution[1]
        )
        self.camera_node = None
        
        # 创建离屏渲染器
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.render_resolution[0],
            viewport_height=self.render_resolution[1]
        )
    
    def spherical_to_cartesian(
        self,
        azimuth: float,
        elevation: float,
        distance: float
    ) -> np.ndarray:
        """将Z-Up球坐标转换为Y-Up笛卡尔坐标"""
        az = azimuth
        el = np.radians(elevation)
        
        y = distance * np.sin(el)
        r_horizontal = distance * np.cos(el)
        x = r_horizontal * np.cos(az)
        z = r_horizontal * np.sin(az)
        
        return np.array([x, y, z]) + self.mesh_center
    
    def get_camera_pose(
        self,
        azimuth: float,
        elevation: float,
        distance: float,
        debug: bool = False
    ) -> np.ndarray:
        """
        计算相机位姿矩阵
        
        参数:
            azimuth: 方位角（弧度）
            elevation: 仰角（度）
            distance: 距离
            debug: 是否输出调试信息
        """
        camera_position = self.spherical_to_cartesian(azimuth, elevation, distance)
        target_position = self.mesh_center
        
        # 计算相机看向向量（从相机指向目标）
        look_direction = target_position - camera_position
        look_direction_normalized = look_direction / np.linalg.norm(look_direction)
        
        if debug:
            print(f"\n=== 相机位置调试信息 ===")
            print(f"方位角: {np.degrees(azimuth):.2f}°, 仰角: {elevation:.2f}°, 距离: {distance:.4f}")
            print(f"相机位置: {camera_position}")
            print(f"模型中心: {target_position}")
            print(f"相机->目标向量: {look_direction}")
            print(f"归一化朝向: {look_direction_normalized}")
        
        # 计算相机变换矩阵（look_at 函数）
        camera_pose = look_at(camera_position, target_position, up=np.array([0.0, 1.0, 0.0]))
        
        return camera_pose
    
    def render_with_alpha(
        self,
        azimuth: float,
        elevation: float,
        distance: float
    ) -> np.ndarray:
        """
        渲染图像（包含alpha通道）
        
        返回：RGBA图像，shape为 (H, W, 4)
        """
        # 获取相机位姿
        camera_pose = self.get_camera_pose(azimuth, elevation, distance)
        
        # 更新场景中的相机
        if self.camera_node is not None:
            self.scene.remove_node(self.camera_node)
        self.camera_node = self.scene.add(self.camera, pose=camera_pose)
        
        # 渲染
        color, depth = self.renderer.render(self.scene)
        
        # 从深度图生成alpha通道（深度>0的像素为不透明）
        alpha = (depth > 0).astype(np.uint8) * 255
        
        # 组合为RGBA
        rgba = np.dstack([color[:, :, :3], alpha])
        
        return rgba
    
    def render_panorama(
        self,
        distance_multiplier: float = CAMERA_DISTANCE,
        azimuth_interval: int = AZIMUTH_INTERVAL,
        elevation_interval: int = ELEVATION_INTERVAL,
        elevation_range: Tuple[float, float] = ELEVATION_RANGE
    ):
        """
        生成全景渲染图像
        
        参数：
            distance_multiplier: 相机距离（相对于网格对角线）
            azimuth_interval: 方位角间隔（度）
            elevation_interval: 仰角间隔（度）
            elevation_range: 仰角范围 (min, max)，单位：度
        """
        distance = distance_multiplier * self.mesh_scale
        
        # 创建输出目录并清空旧文件
        os.makedirs(PANORAMA_OUTPUT_DIR, exist_ok=True)
        
        # 清空输出目录中的旧文件
        print(f"正在清空输出目录: {PANORAMA_OUTPUT_DIR}")
        for filename in os.listdir(PANORAMA_OUTPUT_DIR):
            filepath = os.path.join(PANORAMA_OUTPUT_DIR, filename)
            if os.path.isfile(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"  警告: 无法删除文件 {filename}: {e}")
        print("✓ 输出目录已清空")
        
        # 生成所有角度组合
        azimuths = np.arange(0, 360, azimuth_interval)
        elevations = np.arange(elevation_range[0], elevation_range[1] + elevation_interval, elevation_interval)
        
        total_count = len(azimuths) * len(elevations)
        current_count = 0
        
        print(f"\n开始生成全景渲染...")
        print(f"  距离: {distance_multiplier:.1f}x 网格尺寸 ({distance:.4f} 单位)")
        print(f"  方位角: 0° ~ 360°，间隔 {azimuth_interval}°，共 {len(azimuths)} 个")
        print(f"  仰角: {elevation_range[0]}° ~ {elevation_range[1]}°，间隔 {elevation_interval}°，共 {len(elevations)} 个")
        print(f"  总计: {total_count} 张图像")
        print()
        
        for elevation_deg in elevations:
            for azimuth_deg in azimuths:
                current_count += 1
                
                # 转换为弧度
                azimuth_rad = np.radians(azimuth_deg)
                
                # 渲染
                rgba_image = self.render_with_alpha(
                    azimuth_rad,
                    elevation_deg,
                    distance
                )
                
                # 生成文件名
                filename = f"az_{azimuth_deg:03d}_el_{elevation_deg:+03d}.png"
                filepath = os.path.join(PANORAMA_OUTPUT_DIR, filename)
                
                # 保存（PNG格式支持RGBA）
                # OpenCV需要BGR格式，所以转换
                bgra_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)
                cv2.imwrite(filepath, bgra_image)
                
                # 打印进度
                if current_count % max(1, total_count // 10) == 0:
                    print(f"  已完成: {current_count}/{total_count} ({100*current_count/total_count:.1f}%)")
        
        print(f"\n✓ 全景渲染完成！")
        print(f"  输出目录: {PANORAMA_OUTPUT_DIR}")
        print(f"  图像总数: {total_count}")
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'renderer'):
            self.renderer.delete()


def test_camera_positions(renderer: PanoramaRenderer):
    """测试几个关键角度的相机位置和朝向"""
    print("\n" + "=" * 80)
    print("相机位置和朝向测试")
    print("=" * 80)
    
    distance = CAMERA_DISTANCE * renderer.mesh_scale
    
    # 测试几个关键角度
    test_cases = [
        (0, 0, "0°方位角, 0°仰角 (正前方)"),
        (90, 0, "90°方位角, 0°仰角 (右侧)"),
        (180, 0, "180°方位角, 0°仰角 (后侧)"),
        (270, 0, "270°方位角, 0°仰角 (左侧)"),
        (0, 30, "0°方位角, 30°仰角 (上方)"),
        (0, -30, "0°方位角, -30°仰角 (下方)"),
        (45, 45, "45°方位角, 45°仰角 (对角)"),
    ]
    
    for azimuth_deg, elevation_deg, description in test_cases:
        azimuth_rad = np.radians(azimuth_deg)
        
        # 获取相机位置（带调试输出）
        print(f"\n{description}")
        print("-" * 80)
        
        # 获取相机位置
        camera_pos = renderer.spherical_to_cartesian(azimuth_rad, elevation_deg, distance)
        
        # 计算朝向向量
        look_direction = renderer.mesh_center - camera_pos
        look_direction_norm = np.linalg.norm(look_direction)
        look_direction_normalized = look_direction / look_direction_norm
        
        # 相对位置向量（从模型中心指向相机）
        relative_pos = camera_pos - renderer.mesh_center
        relative_distance = np.linalg.norm(relative_pos)
        relative_pos_normalized = relative_pos / relative_distance
        
        print(f"相机位置: ({camera_pos[0]:+8.4f}, {camera_pos[1]:+8.4f}, {camera_pos[2]:+8.4f})")
        print(f"模型中心: ({renderer.mesh_center[0]:+8.4f}, {renderer.mesh_center[1]:+8.4f}, {renderer.mesh_center[2]:+8.4f})")
        print(f"相对位置: ({relative_pos[0]:+8.4f}, {relative_pos[1]:+8.4f}, {relative_pos[2]:+8.4f})")
        print(f"相对距离: {relative_distance:.4f} (应该为 {distance:.4f})")
        print(f"相对位置(归一化): ({relative_pos_normalized[0]:+8.4f}, {relative_pos_normalized[1]:+8.4f}, {relative_pos_normalized[2]:+8.4f})")
        print(f"相机朝向(指向目标): ({look_direction_normalized[0]:+8.4f}, {look_direction_normalized[1]:+8.4f}, {look_direction_normalized[2]:+8.4f})")
        
        # 验证朝向是否指向模型中心（应该与相对位置相反）
        expected_direction = -relative_pos_normalized
        direction_match = np.allclose(look_direction_normalized, expected_direction)
        print(f"朝向验证: {'✓ 正确' if direction_match else '✗ 错误'}")


def main():
    """主函数"""
    print("=" * 60)
    print("全景渲染脚本")
    print("=" * 60)
    
    # 检查网格文件
    if not os.path.exists(MESH_PATH):
        print(f"[错误] 网格文件不存在: {MESH_PATH}")
        return
    
    # 创建渲染器
    renderer = PanoramaRenderer(
        mesh_path=MESH_PATH,
        render_resolution=(RENDER_WIDTH, RENDER_HEIGHT),
        fov=CAMERA_FOV
    )
    
    # 先测试几个关键角度的相机位置
    test_camera_positions(renderer)
    
    # 生成全景
    renderer.render_panorama(
        distance_multiplier=CAMERA_DISTANCE,
        azimuth_interval=AZIMUTH_INTERVAL,
        elevation_interval=ELEVATION_INTERVAL,
        elevation_range=ELEVATION_RANGE
    )
    
    # 清理
    renderer.cleanup()
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
