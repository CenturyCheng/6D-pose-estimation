"""
基于分析合成的6D位姿估计
=============================================
使用CMA-ES优化算法，通过将3D网格的轮廓与目标2D二值掩码对齐，
来估计相机参数（方位角、仰角、距离）。

依赖: pip install pyrender trimesh cma opencv-python numpy matplotlib
"""

import numpy as np
import trimesh
import pyrender
import cv2
import cma
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
import os
import shutil

# ============================================================================
# 可调整参数配置区
# ============================================================================

# --- 文件路径配置 ---
MESH_PATH = "./model.glb"           # 3D网格文件路径（支持 .obj, .stl, .ply 等）
TARGET_MASK_PATH = "./target.png"   # 目标二值掩码图像路径
OUTPUT_DIR = "./output"             # 输出结果保存目录

# --- 渲染参数 ---
RENDER_WIDTH = 256                  # 渲染图像宽度（像素）
RENDER_HEIGHT = 256                 # 渲染图像高度（像素）
CAMERA_FOV = 45.0                   # 相机视场角（度）

# --- CMA-ES优化参数 ---
CMAES_SIGMA = 5.0                   # CMA-ES初始步长（标准差）
CMAES_POPSIZE = 50                  # 种群大小（每代评估的个体数）
CMAES_MAXITER = 300                 # 最大迭代次数
CMAES_TOLX = 1e-6                   # 参数变化容差（收敛条件）
CMAES_TOLFUN = 1e-8                 # 目标函数值容差（收敛条件）

# --- 参数搜索范围 ---
AZIMUTH_RANGE = (0.0, 2 * np.pi)    # 方位角范围（弧度），0~2π
ELEVATION_RANGE = (-89.0, 89.0)     # 仰角范围（度），避免万向锁
DISTANCE_MIN = 0.8                 # 最小距离（相对于网格尺寸的倍数）
DISTANCE_MAX = 5.0                  # 最大距离（相对于网格尺寸的倍数）

# --- 其他设置 ---
VERBOSE = True                      # 是否打印详细信息
SAVE_VISUALIZATION = True           # 是否保存可视化结果
SHOW_INTERACTIVE = False            # 是否显示交互式pyrender窗口
DEBUG_OUTPUT = False                 # 是否输出调试渲染图像

# ============================================================================
# 设置无头渲染环境（服务器环境必需）
# ============================================================================
# os.environ['PYOPENGL_PLATFORM'] = 'egl'


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """
    标准 OpenGL/pyrender LookAt 矩阵计算（Y-Up 世界坐标系）。
    
    参数:
        eye: 相机位置
        target: 相机看向的目标点
        up: 上方向向量，默认为 [0, 1, 0]（Y-Up）
    
    返回:
        4x4 相机变换矩阵
    """
    if up is None:
        # Y-Up 坐标系中，全局上方是 [0, 1, 0]
        up = np.array([0.0, 1.0, 0.0])

    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    # 标准 OpenGL LookAt 实现
    z_axis = eye - target
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        # 处理万向锁：up 和 z_axis 平行
        x_axis = np.cross(np.array([1.0, 0.0, 0.0]), z_axis)
    
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # 构建矩阵
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = x_axis  # 相机的 +X 轴（右方向）
    matrix[:3, 1] = y_axis  # 相机的 +Y 轴（上方向）
    matrix[:3, 2] = z_axis  # 相机的 +Z 轴（后方向）
    matrix[:3, 3] = eye     # 相机位置

    return matrix


class PoseOptimizer:
    """
    位姿优化器：通过分析合成方法估计相机位姿。
    
    核心特性:
    - 网格几何体保持原样不修改
    - 相机位置相对于网格包围盒中心计算
    - 使用CMA-ES进行全局优化
    - 支持方位角的周期性处理
    """
    
    def __init__(
        self,
        mesh_path: str,
        render_resolution: Tuple[int, int] = (RENDER_WIDTH, RENDER_HEIGHT),
        fov: float = CAMERA_FOV,
        verbose: bool = VERBOSE
    ):
        """
        初始化位姿优化器。
        
        参数:
            mesh_path: 网格文件路径
            render_resolution: 渲染分辨率 (宽, 高)
            fov: 相机视场角（度）
            verbose: 是否输出详细信息
        """
        self.render_resolution = render_resolution
        self.fov = fov
        self.verbose = verbose
        self.iteration_count = 0
        self.best_iou = 0.0
        self.history = []  # 记录优化历史
        
        # 加载网格（不修改顶点！）
        if self.verbose:
            print(f"正在加载网格: {mesh_path}")
        
        self.mesh = trimesh.load(mesh_path, force='mesh')
        
        # 注意：不在这里修改网格，而是在计算摄像机位置时处理坐标系转换
        # 这样可以保持模型原始几何不变
        
        # 计算旋转后的几何中心（作为 LookAt 的目标点）
        self.mesh_center = self.mesh.bounding_box.centroid.copy()
        
        # 计算网格尺寸，用于确定合适的相机距离
        bounds = self.mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        self.mesh_extent = bounds[1] - bounds[0]  # 包围盒尺寸
        self.mesh_scale = np.linalg.norm(self.mesh_extent)  # 对角线长度
        
        if self.verbose:
            print(f"1. 边界范围 (Bounds):")
            print(f"   Min (左下后): {bounds[0]}")
            print(f"   Max (右上前): {bounds[1]}")
            print(f"  网格中心: {self.mesh_center}")
            print(f"  网格尺寸: {self.mesh_extent}")
            print(f"  对角线长度: {self.mesh_scale:.4f}")
        
        # 初始化pyrender场景
        self._setup_scene()
        
        if self.verbose:
            print("初始化完成！")
    
    def _setup_scene(self):
        """设置pyrender渲染场景。"""
        # 创建场景
        self.scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0],  # 黑色背景
            ambient_light=[0.3, 0.3, 0.3]   # 环境光
        )
        
        # 将trimesh转换为pyrender网格并添加到场景
        # 使用简单的白色材质，便于生成轮廓
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0
        )
        pyrender_mesh = pyrender.Mesh.from_trimesh(self.mesh, material=material)
        self.mesh_node = self.scene.add(pyrender_mesh)
        
        # 添加方向光
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        self.scene.add(light, pose=np.eye(4))
        
        # 创建相机（暂时不添加到场景，渲染时动态设置）
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
        """
        将Z-Up球坐标转换为Y-Up笛卡尔坐标。
        
        输入参数定义在Z-Up系统中：
            azimuth: 绕Z轴旋转角度（弧度）
            elevation: 从XY平面向上的角度（度）
            distance: 到原点的距离
        
        直接在Y-Up空间中计算，但遵循Z-Up的几何含义。
        
        返回:
            [x, y, z] 笛卡尔坐标（Y-Up 坐标系中的位置）
        """
        az = azimuth
        el = np.radians(elevation)
        
        # Z-Up定义的球坐标 → Y-Up笛卡尔坐标的直接映射
        # 在Z-Up中: x = r*cos(el)*cos(az), y = r*cos(el)*sin(az), z = r*sin(el)
        # 在Y-Up中: x = r*cos(el)*cos(az), y = r*sin(el), z = r*cos(el)*sin(az)
        # (这样Z-Up中的Z轴高度对应Y-Up中的Y轴高度)
        
        y = distance * np.sin(el)  # 高度（Y-Up中为Y轴）
        r_horizontal = distance * np.cos(el)  # 水平平面半径
        x = r_horizontal * np.cos(az)
        z = r_horizontal * np.sin(az)
        
        # 加上中心点偏移
        return np.array([x, y, z]) + self.mesh_center
    
    def get_camera_pose(
        self, 
        azimuth: float, 
        elevation: float, 
        distance: float
    ) -> np.ndarray:
        """
        计算给定参数下的相机位姿矩阵（Y-Up 坐标系）。
        
        相机环绕网格中心，始终看向网格中心。
        
        参数:
            azimuth: 方位角（弧度）
            elevation: 仰角（度）
            distance: 相机到网格中心的距离
        
        返回:
            4x4 相机变换矩阵（Y-Up 坐标系）
        """
        # 使用 Y-Up 逻辑计算相机位置
        camera_position = self.spherical_to_cartesian(azimuth, elevation, distance)
        
        # 目标位置 = 网格中心
        target_position = self.mesh_center
        
        # 标准 LookAt（Y-Up，up 向量是 [0,1,0]）
        camera_pose = look_at(camera_position, target_position, up=np.array([0.0, 1.0, 0.0]))
        
        return camera_pose
    
    def compute_adaptive_fov(
        self,
        azimuth: float,
        elevation: float,
        distance: float,
        margin: float = 0.05
    ) -> float:
        """
        计算自适应FOV，使得模型包围盒在图像中尽可能占满。
        
        原理：
        - 计算相机到网格中心的距离
        - 计算包围盒在相机视图中的投影尺寸
        - 根据投影尺寸和距离计算所需的FOV
        - 添加5%的余量
        
        参数:
            azimuth: 方位角（弧度）
            elevation: 仰角（度）
            distance: 相机到网格中心的距离
            margin: 余量比例（默认5%）
        
        返回:
            自适应FOV（度）
        """
        # 获取相机位置
        camera_position = self.spherical_to_cartesian(azimuth, elevation, distance)
        
        # 计算从相机到网格中心的向量
        to_center = self.mesh_center - camera_position
        dist_to_center = np.linalg.norm(to_center)
        
        # 计算相机视图方向（归一化）
        view_dir = to_center / dist_to_center
        
        # 计算相机右方向和上方向
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(view_dir, up)
        if np.linalg.norm(right) < 1e-6:
            # 处理万向锁情况
            right = np.cross(view_dir, np.array([1.0, 0.0, 0.0]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, view_dir)
        up = up / np.linalg.norm(up)
        
        # 获取包围盒的8个顶点
        bounds = self.mesh.bounds
        corners = np.array([
            [bounds[0, 0], bounds[0, 1], bounds[0, 2]],
            [bounds[1, 0], bounds[0, 1], bounds[0, 2]],
            [bounds[0, 0], bounds[1, 1], bounds[0, 2]],
            [bounds[1, 0], bounds[1, 1], bounds[0, 2]],
            [bounds[0, 0], bounds[0, 1], bounds[1, 2]],
            [bounds[1, 0], bounds[0, 1], bounds[1, 2]],
            [bounds[0, 0], bounds[1, 1], bounds[1, 2]],
            [bounds[1, 0], bounds[1, 1], bounds[1, 2]],
        ])
        
        # 计算每个顶点在相机视图中的投影
        max_horizontal = 0.0
        max_vertical = 0.0
        
        for corner in corners:
            # 从相机到顶点的向量
            to_corner = corner - camera_position
            
            # 投影到相机视图平面
            # 水平方向投影（右方向）
            horizontal_proj = np.dot(to_corner, right)
            # 垂直方向投影（上方向）
            vertical_proj = np.dot(to_corner, up)
            
            # 计算视角（使用到中心的距离作为参考，避免近大远小的影响）
            # 这样所有顶点都使用相同的参考距离，确保FOV计算准确
            horizontal_angle = np.abs(np.arctan2(horizontal_proj, dist_to_center))
            vertical_angle = np.abs(np.arctan2(vertical_proj, dist_to_center))
            
            max_horizontal = max(max_horizontal, horizontal_angle)
            max_vertical = max(max_vertical, vertical_angle)
        
        # 计算所需的FOV（添加余量）
        # FOV = 2 * max_angle * (1 + margin)
        fov_horizontal = 2 * max_horizontal * (1 + margin)
        fov_vertical = 2 * max_vertical * (1 + margin)
        
        # pyrender使用垂直FOV (yfov)，所以需要将水平FOV转换为等效的垂直FOV
        aspect_ratio = self.render_resolution[0] / self.render_resolution[1]
        
        # 将水平FOV转换为等效的垂直FOV
        # 公式: v_fov = 2 * arctan(tan(h_fov/2) / aspect_ratio)
        fov_vertical_from_horizontal = 2 * np.arctan(np.tan(fov_horizontal / 2) / aspect_ratio)
        
        # 取两者中较大的那个，确保模型完全可见
        fov = max(fov_vertical, fov_vertical_from_horizontal)
        
        # 转换为度数
        fov_deg = np.degrees(fov)
        
        # 限制FOV在合理范围内（避免过小或过大）
        fov_deg = np.clip(fov_deg, 1.0, 120.0)
        
        return fov_deg
    
    def render_silhouette(
        self, 
        azimuth: float, 
        elevation: float, 
        distance: float,
        use_adaptive_fov: bool = True
    ) -> np.ndarray:
        """
        渲染给定相机参数下的网格轮廓。
        
        参数:
            azimuth: 方位角（弧度）
            elevation: 仰角（度）
            distance: 距离
            use_adaptive_fov: 是否使用自适应FOV（默认True）
        
        返回:
            二值轮廓图像 (H, W)，值为0或255
        """
        # 获取相机位姿
        camera_pose = self.get_camera_pose(azimuth, elevation, distance)
        
        # 计算自适应FOV
        if use_adaptive_fov:
            adaptive_fov = self.compute_adaptive_fov(azimuth, elevation, distance, margin=0.05)
            # 创建新的相机实例，避免修改全局相机FOV
            camera = pyrender.PerspectiveCamera(
                yfov=np.radians(adaptive_fov),
                aspectRatio=self.render_resolution[0] / self.render_resolution[1]
            )
        else:
            camera = self.camera
        
        # 更新场景中的相机
        if self.camera_node is not None:
            self.scene.remove_node(self.camera_node)
        self.camera_node = self.scene.add(camera, pose=camera_pose)
        
        # 渲染
        color, depth = self.renderer.render(self.scene)
        
        # 从深度图生成轮廓（深度>0的像素为前景）
        silhouette = (depth > 0).astype(np.uint8) * 255
        
        return silhouette
    
    def compute_mask_center(self, mask: np.ndarray) -> np.ndarray:
        """
        计算mask的2D中心（基于非零像素的质心）。
        
        参数:
            mask: 二值掩码
        
        返回:
            [x, y] 中心坐标
        """
        # 获取非零像素的坐标
        y_coords, x_coords = np.where(mask > 1)
        
        if len(x_coords) == 0:
            # 如果mask为空，返回图像中心
            return np.array([mask.shape[1] / 2, mask.shape[0] / 2])
        
        # 计算质心
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        return np.array([center_x, center_y])
    
    def compute_mask_bbox(self, mask: np.ndarray) -> Tuple[float, float, float, float]:
        """
        计算mask的AABB包围盒。
        
        参数:
            mask: 二值掩码
        
        返回:
            (min_x, min_y, max_x, max_y) 包围盒坐标
        """
        # 获取非零像素的坐标
        y_coords, x_coords = np.where(mask > 127)
        
        if len(x_coords) == 0:
            # 如果mask为空，返回整个图像
            return (0, 0, mask.shape[1], mask.shape[0])
        
        min_x = np.min(x_coords)
        max_x = np.max(x_coords)
        min_y = np.min(y_coords)
        max_y = np.max(y_coords)
        
        return (min_x, min_y, max_x, max_y)
    
    def align_masks(
        self,
        rendered_mask: np.ndarray,
        target_mask: np.ndarray
    ) -> np.ndarray:
        """
        将渲染mask对齐到target mask（平移+缩放）。
        
        对齐步骤：
        1. 计算两个mask的2D中心
        2. 计算两个mask的AABB框尺寸
        3. 计算缩放比例（基于AABB框的xyscale差的平均值）
        4. 平移渲染mask使其中心与target mask中心对齐
        5. 缩放渲染mask使其尺寸与target mask匹配
        
        参数:
            rendered_mask: 渲染的掩码
            target_mask: 目标掩码
        
        返回:
            对齐后的渲染掩码
        """
        # 计算两个mask的中心
        rendered_center = self.compute_mask_center(rendered_mask)
        target_center = self.compute_mask_center(target_mask)
        
        # 计算平移向量
        translation = target_center - rendered_center
        
        # 计算两个mask的AABB框
        rendered_bbox = self.compute_mask_bbox(rendered_mask)
        target_bbox = self.compute_mask_bbox(target_mask)
        
        # 计算AABB框的宽度和高度
        rendered_width = rendered_bbox[2] - rendered_bbox[0]
        rendered_height = rendered_bbox[3] - rendered_bbox[1]
        target_width = target_bbox[2] - target_bbox[0]
        target_height = target_bbox[3] - target_bbox[1]
        
        # 计算缩放比例（基于xyscale差的平均值）
        if rendered_width > 0 and rendered_height > 0:
            scale_x = target_width / rendered_width
            scale_y = target_height / rendered_height
            scale = (scale_x + scale_y) / 2  # 平均值
        else:
            scale = 1.0
        
        # 创建变换矩阵（平移+缩放）
        # 先平移到原点，缩放，再平移到目标位置
        M = np.array([
            [scale, 0, target_center[0] - rendered_center[0] * scale],
            [0, scale, target_center[1] - rendered_center[1] * scale]
        ], dtype=np.float32)
        
        # 应用变换
        aligned_mask = cv2.warpAffine(
            rendered_mask,
            M,
            (target_mask.shape[1], target_mask.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # 二值化
        _, aligned_mask = cv2.threshold(aligned_mask, 1, 255, cv2.THRESH_BINARY)
        
        return aligned_mask
    
    def compute_iou(self, mask1: np.ndarray, mask2: np.ndarray, align: bool = True) -> float:
        """
        计算两个二值掩码之间的IoU（交并比）。
        
        参数:
            mask1: 第一个掩码
            mask2: 第二个掩码
            align: 是否在计算IoU前对齐mask（默认True）
        
        返回:
            IoU值，范围[0, 1]
        """
        # 如果需要对齐，将mask1对齐到mask2
        if align:
            mask1 = self.align_masks(mask1, mask2)
        
        # 确保是二值掩码
        m1 = (mask1 > 127).astype(bool)
        m2 = (mask2 > 127).astype(bool)
        
        # 计算交集和并集
        intersection = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        
        if union == 0:
            return 0.0
        
        return float(intersection) / float(union)
    
    def _normalize_params(self, params: np.ndarray) -> Tuple[float, float, float]:
        """
        规范化参数：处理方位角周期性，限制仰角和距离范围。
        
        参数:
            params: [azimuth, elevation, distance] 原始参数
        
        返回:
            (azimuth, elevation, distance) 规范化后的参数
        """
        azimuth, elevation, distance = params
        
        # 方位角：周期性处理（0到2π）
        azimuth = azimuth % (2 * np.pi)
        
        # 仰角：限制范围，避免万向锁
        elevation = np.clip(elevation, ELEVATION_RANGE[0], ELEVATION_RANGE[1])
        
        # 距离：必须为正值
        min_dist = DISTANCE_MIN * self.mesh_scale
        max_dist = DISTANCE_MAX * self.mesh_scale
        distance = np.clip(distance, min_dist, max_dist)
        
        return azimuth, elevation, distance
    
    def _objective_function(self, params: np.ndarray, target_mask: np.ndarray) -> float:
        """
        CMA-ES的目标函数：返回负IoU（因为CMA-ES最小化目标函数）。
        
        参数:
            params: [azimuth, elevation, distance]
            target_mask: 目标掩码
        
        返回:
            负IoU值
        """
        # 规范化参数
        azimuth, elevation, distance = self._normalize_params(params)
        
        # 渲染轮廓
        rendered = self.render_silhouette(azimuth, elevation, distance)
        
        # 调整尺寸以匹配目标掩码
        if rendered.shape != target_mask.shape:
            rendered = cv2.resize(
                rendered, 
                (target_mask.shape[1], target_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # 计算IoU（自动对齐）
        iou = self.compute_iou(rendered, target_mask, align=True)
        
        # 获取对齐后的渲染mask用于显示
        rendered_aligned = self.align_masks(rendered, target_mask)
        
        DEBUG_RENDER_PROB = 0.02  # 2% 的概率输出
        if np.random.random() < DEBUG_RENDER_PROB and DEBUG_OUTPUT:
            debug_dir = os.path.join(OUTPUT_DIR, 'debug_renders')
            os.makedirs(debug_dir, exist_ok=True)
            
            # 创建对比图
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # 目标掩码
            axes[0].imshow(target_mask, cmap='gray')
            axes[0].set_title('Target')
            axes[0].axis('off')
            
            # 对齐后的渲染
            axes[1].imshow(rendered_aligned, cmap='gray')
            rel_dist = elevation / self.mesh_scale
            axes[1].set_title(f'Rendered (Aligned)\nAz:{np.degrees(azimuth):.1f}° El:{elevation:.1f}° D:{distance/self.mesh_scale:.2f}x')
            axes[1].axis('off')
            
            # 叠加对比（使用对齐后的图像）
            overlay = np.zeros((*target_mask.shape, 3), dtype=np.uint8)
            overlay[:, :, 0] = target_mask  # 红色 = 目标
            overlay[:, :, 1] = rendered_aligned     # 绿色 = 渲染（对齐后）
            axes[2].imshow(overlay)
            axes[2].set_title(f'Overlay (IoU: {iou:.4f})')
            axes[2].axis('off')
            
            plt.suptitle(f'Iteration {self.iteration_count}', fontsize=12)
            plt.tight_layout()
            
            save_path = os.path.join(debug_dir, f'iter_{self.iteration_count:05d}_iou_{iou:.4f}.png')
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            if self.verbose:
                print(f"    [Debug] Saved render: {save_path}")

        # 记录最佳结果
        if iou > self.best_iou:
            self.best_iou = iou
            self.best_params = (azimuth, elevation, distance)
        
        self.iteration_count += 1
        
        # 记录历史
        self.history.append({
            'iteration': self.iteration_count,
            'iou': iou,
            'azimuth': azimuth,
            'elevation': elevation,
            'distance': distance
        })
        
        # 定期打印进度
        if self.verbose and self.iteration_count % 20 == 0:
            print(f"  迭代 {self.iteration_count}: IoU = {iou:.4f}, "
                  f"最佳 IoU = {self.best_iou:.4f}")
        
        # 返回负IoU（因为CMA-ES最小化目标函数）
        return -iou
    
    def optimize(
        self,
        target_mask: np.ndarray,
        initial_params: Optional[np.ndarray] = None,
        sigma: float = CMAES_SIGMA,
        popsize: int = CMAES_POPSIZE,
        maxiter: int = CMAES_MAXITER
    ) -> Dict[str, Any]:
        """
        执行CMA-ES优化，寻找最佳相机参数。
        
        参数:
            target_mask: 目标二值掩码图像
            initial_params: 初始参数 [azimuth, elevation, distance]，None则随机初始化
            sigma: CMA-ES初始步长
            popsize: 种群大小
            maxiter: 最大迭代次数
        
        返回:
            包含优化结果的字典
        """
        # 清理 debug 输出目录
        if DEBUG_OUTPUT:
            debug_dir = os.path.join(OUTPUT_DIR, 'debug_renders')
            if os.path.exists(debug_dir):
                try:
                    shutil.rmtree(debug_dir)
                    if self.verbose:
                        print(f"已清理旧的调试输出目录: {debug_dir}")
                except Exception as e:
                    print(f"清理调试目录时出错: {e}")

        # 重置状态
        self.iteration_count = 0
        self.best_iou = 0.0
        self.best_params = None
        self.history = []
        
        # 预处理目标掩码
        if len(target_mask.shape) == 3:
            target_mask = cv2.cvtColor(target_mask, cv2.COLOR_BGR2GRAY)
        _, target_mask = cv2.threshold(target_mask, 1, 255, cv2.THRESH_BINARY)
        
        # 调整目标掩码大小
        target_mask = cv2.resize(
            target_mask,
            self.render_resolution,
            interpolation=cv2.INTER_NEAREST
        )
        
        if self.verbose:
            print(f"\n开始CMA-ES优化...")
            print(f"  种群大小: {popsize}")
            print(f"  最大迭代: {maxiter}")
            print(f"  初始步长: {sigma}")
        
        # 初始参数
        if initial_params is None:
            # 随机初始化
            initial_params = np.array([
                np.random.uniform(0, 2 * np.pi),           # azimuth
                np.random.uniform(-30, 30),                 # elevation
                self.mesh_scale * 2.5                       # distance
            ])
        
        if self.verbose:
            print(f"  初始参数: 方位角={np.degrees(initial_params[0]):.1f}°, "
                  f"仰角={initial_params[1]:.1f}°, "
                  f"距离={initial_params[2]:.2f}")
        
        # CMA-ES优化选项
        opts = {
            'popsize': popsize,
            'maxiter': maxiter,
            'tolx': CMAES_TOLX,
            'tolfun': CMAES_TOLFUN,
            'verb_disp': 0,  # 禁用CMA-ES的内置输出
            'bounds': [
                [0, ELEVATION_RANGE[0], DISTANCE_MIN * self.mesh_scale],      # 下界
                [2 * np.pi, ELEVATION_RANGE[1], DISTANCE_MAX * self.mesh_scale]  # 上界
            ]
        }
        
        # 执行CMA-ES优化
        es = cma.CMAEvolutionStrategy(initial_params, sigma, opts)
        
        while not es.stop():
            # 获取候选解
            solutions = es.ask()
            
            # 评估每个候选解
            fitnesses = [
                self._objective_function(sol, target_mask) 
                for sol in solutions
            ]
            
            # 更新CMA-ES
            es.tell(solutions, fitnesses)
        
        # 获取最终结果
        final_params = self._normalize_params(es.result.xbest)
        
        if self.verbose:
            print(f"\n优化完成!")
            print(f"  总评估次数: {self.iteration_count}")
            print(f"  最佳IoU: {self.best_iou:.4f}")
            print(f"  最佳参数 (Y-Up 系统):")
            print(f"    方位角: {np.degrees(final_params[0]):.2f}°")
            print(f"    仰角: {final_params[1]:.2f}°")
            print(f"    距离: {final_params[2]:.4f}")
        
        # 转换为 Z-Up 参数
        zup_params = self.get_final_zup_params(final_params[0], final_params[1], final_params[2])
        
        # 获取 Y-Up 坐标系的相机位姿（用于渲染验证）
        camera_pose_yup = self.get_camera_pose(*final_params)
        
        if self.verbose:
            print(f"\n  转换为 Z-Up 参数:")
            print(f"    方位角: {zup_params['azimuth']:.2f}°")
            print(f"    仰角: {zup_params['elevation']:.2f}°")
            print(f"    距离: {zup_params['distance']:.4f}")
        
        return {
            'azimuth': zup_params['azimuth'],
            'azimuth_deg': zup_params['azimuth'],
            'elevation': zup_params['elevation'],
            'distance': zup_params['distance'],
            'iou': self.best_iou,
            'iterations': self.iteration_count,
            'history': self.history,
            'camera_pose': camera_pose_yup
        }
    
    def visualize_result(
        self,
        target_mask: np.ndarray,
        result: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """
        可视化优化结果。
        
        参数:
            target_mask: 目标掩码
            result: optimize()返回的结果字典
            save_path: 保存路径，None则显示
        """
        # 渲染最终结果（注意：result['azimuth']是度数，需要转换为弧度）
        rendered = self.render_silhouette(
            np.radians(result['azimuth']),  # 转换为弧度
            result['elevation'],
            result['distance']
        )
        
        # 调整尺寸
        if len(target_mask.shape) == 3:
            target_mask = cv2.cvtColor(target_mask, cv2.COLOR_BGR2GRAY)
        target_mask = cv2.resize(target_mask, self.render_resolution, interpolation=cv2.INTER_NEAREST)
        
        # 对齐渲染mask到target mask
        rendered_aligned = self.align_masks(rendered, target_mask)
        
        # 创建对比图
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # 目标掩码
        axes[0].imshow(target_mask, cmap='gray')
        axes[0].set_title('Target')
        axes[0].axis('off')
        
        # 对齐后的渲染结果
        axes[1].imshow(rendered_aligned, cmap='gray')
        axes[1].set_title(f'Rendered (Aligned)\nAz: {result["azimuth_deg"]:.1f}° El: {result["elevation"]:.1f}°\n'
                         f'D: {result["distance"]/self.mesh_scale:.2f}x')
        axes[1].axis('off')
        
        # 叠加对比（红=目标，绿=渲染，黄=重叠）
        overlay = np.zeros((*target_mask.shape, 3), dtype=np.uint8)
        overlay[:, :, 0] = target_mask  # 红色通道 = 目标
        overlay[:, :, 1] = rendered_aligned     # 绿色通道 = 渲染（对齐后）
        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay\nRed=Target, Green=Rendered\nIoU: {result["iou"]:.4f}')
        axes[2].axis('off')
        
        # 优化历史
        if result['history']:
            ious = [h['iou'] for h in result['history']]
            axes[3].plot(ious, 'b-', alpha=0.3)
            # 添加滑动平均线
            window = min(20, len(ious))
            if window > 1:
                smoothed = np.convolve(ious, np.ones(window)/window, mode='valid')
                axes[3].plot(range(window-1, len(ious)), smoothed, 'r-', linewidth=2)
            axes[3].set_xlabel('Evaluations')
            axes[3].set_ylabel('IoU')
            axes[3].set_title('Optimization Progress')
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', 
                       exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"结果已保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def show_interactive_result(self, result: Dict[str, Any]):
        """
        使用pyrender.Viewer显示交互式可视化结果。
        
        可以用鼠标在窗口中：
        - 左键拖动：旋转视图
        - 右键拖动：平移视图  
        - 滚轮：缩放
        
        参数:
            result: optimize()返回的结果字典
        """
        # 创建新的场景用于显示（包含网格和相机）
        display_scene = pyrender.Scene(
            bg_color=[0.1, 0.1, 0.1, 1.0],
            ambient_light=[0.5, 0.5, 0.5]
        )
        
        # 添加网格
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.8, 0.8, 0.8, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.8
        )
        pyrender_mesh = pyrender.Mesh.from_trimesh(self.mesh, material=material)
        display_scene.add(pyrender_mesh)
        
        # 添加坐标轴（轴长为网格对角线的1/3）
        axis_length = self.mesh_scale / 3
        axis_mesh = trimesh.creation.axis(axis_length)
        axis_pyrender = pyrender.Mesh.from_trimesh(axis_mesh, smooth=False)
        display_scene.add(axis_pyrender)
        
        # 添加灯光
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        display_scene.add(light, pose=np.eye(4))
        
        # 添加最优相机位置
        camera = pyrender.PerspectiveCamera(
            yfov=np.radians(self.fov),
            aspectRatio=self.render_resolution[0] / self.render_resolution[1]
        )
        display_scene.add(camera, pose=result['camera_pose'])
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("打开交互式可视化窗口...")
            print("=" * 60)
            print("鼠标控制:")
            print("  - 左键拖动: 旋转")
            print("  - 右键拖动: 平移")
            print("  - 滚轮: 缩放")
            print("\n坐标轴说明 (原始模型坐标系):")
            print("  - 红轴: X轴")
            print("  - 绿轴: Y轴")
            print("  - 蓝轴: Z轴（原始模型的向上方向）")
            print("\n关闭窗口以继续...")
            print("=" * 60)
        
        # 显示交互式viewer
        pyrender.Viewer(display_scene)
    
    def cleanup(self):
        """清理渲染资源。"""
        if hasattr(self, 'renderer'):
            self.renderer.delete()
    
    def get_final_zup_params(self, best_azimuth: float, best_elevation: float, best_distance: float) -> Dict[str, Any]:
        """
        因为我们直接用Z-Up参数计算Y-Up位置，所以这里直接返回输入参数。
        
        参数:
            best_azimuth: 方位角（弧度，Z-Up定义）
            best_elevation: 仰角（度，Z-Up定义）
            best_distance: 距离
        
        返回:
            包含 Z-Up 参数的字典
        """
        # 直接使用输入参数（因为我们用的就是Z-Up定义）
        final_azimuth_deg = np.degrees(best_azimuth)
        
        return {
            "azimuth": final_azimuth_deg,
            "elevation": best_elevation,
            "distance": best_distance,
            "azimuth_rad": best_azimuth
        }
    
    def get_relative_distance(self, distance: float) -> float:
        """
        计算相对距离（相对于网格包围盒对角线）。
        
        相对距离 = 绝对距离 / 网格对角线长度
        例如：relative_distance = 2.5 表示相机距离网格中心 2.5 倍网格大小
        
        参数:
            distance: 绝对距离（坐标单位）
        
        返回:
            相对距离系数
        """
        return distance / self.mesh_scale


def create_test_mask(optimizer: PoseOptimizer, azimuth_deg: float = 45.0,
                     elevation_deg: float = 30.0) -> np.ndarray:
    """
    创建测试用的目标掩码（用于自检模式）。
    
    参数:
        optimizer: PoseOptimizer实例
        azimuth_deg: 真实方位角（度）
        elevation_deg: 真实仰角（度）
    
    返回:
        生成的目标掩码
    """
    distance = optimizer.mesh_scale * 2.5
    azimuth_rad = np.radians(azimuth_deg)
    
    mask = optimizer.render_silhouette(azimuth_rad, elevation_deg, distance)
    
    return mask


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数：执行完整的位姿估计流程。
    """
    print("=" * 60)
    print("基于分析合成的6D位姿估计")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 用于记录真实参数（自检模式）
    run_self_check = False
    gt_params = None
    
    # --- 自检模式：如果目标掩码不存在，则生成一个 ---
    if not os.path.exists(TARGET_MASK_PATH):
        print(f"\n[自检模式] 未找到目标掩码: {TARGET_MASK_PATH}")
        print("将使用已知参数生成合成目标掩码进行测试...\n")
        
        # 检查网格文件是否存在
        if not os.path.exists(MESH_PATH):
            print(f"[错误] 网格文件不存在: {MESH_PATH}")
            print("请提供有效的网格文件路径。")
            
            # 创建一个简单的测试网格
            print("\n[自检模式] 创建测试立方体网格...")
            test_mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
            os.makedirs(os.path.dirname(MESH_PATH) if os.path.dirname(MESH_PATH) else '.', exist_ok=True)
            test_mesh.export(MESH_PATH)
            print(f"已创建测试网格: {MESH_PATH}")
        
        # 临时创建优化器来生成目标掩码
        temp_optimizer = PoseOptimizer(
            mesh_path=MESH_PATH,
            render_resolution=(RENDER_WIDTH, RENDER_HEIGHT),
            fov=CAMERA_FOV,
            verbose=False
        )
        
        # 使用已知参数生成目标掩码
        gt_params = {
            'azimuth': np.pi / 4,                      # 45度
            'elevation': 30.0,                          # 30度
            'distance': 2.5 * temp_optimizer.mesh_scale # 2.5倍网格尺寸
        }
        
        print(f"真实参数: 方位角={np.degrees(gt_params['azimuth']):.1f}°, "
              f"仰角={gt_params['elevation']:.1f}°, "
              f"距离={gt_params['distance']:.4f}")
        
        # 使用 render_silhouette 而不是 render_mask
        target_mask = temp_optimizer.render_silhouette(
            gt_params['azimuth'],
            gt_params['elevation'],
            gt_params['distance']
        )
        
        # 保存目标掩码
        os.makedirs(os.path.dirname(TARGET_MASK_PATH) if os.path.dirname(TARGET_MASK_PATH) else '.', exist_ok=True)
        cv2.imwrite(TARGET_MASK_PATH, target_mask)
        print(f"已生成目标掩码: {TARGET_MASK_PATH}\n")
        
        temp_optimizer.cleanup()
        del temp_optimizer
        
        run_self_check = True
    
    # --- 加载目标掩码 ---
    print(f"加载目标掩码: {TARGET_MASK_PATH}")
    target_mask = cv2.imread(TARGET_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    
    if target_mask is None:
        print(f"[错误] 无法加载目标掩码: {TARGET_MASK_PATH}")
        return None, None
    
    print(f"目标掩码尺寸: {target_mask.shape}")
    print(f"目标掩码非零像素: {np.sum(target_mask > 0)}")
    
    # --- 创建优化器 ---
    print(f"\n加载网格: {MESH_PATH}")
    optimizer = PoseOptimizer(
        mesh_path=MESH_PATH,
        render_resolution=(RENDER_WIDTH, RENDER_HEIGHT),
        fov=CAMERA_FOV,
        verbose=VERBOSE
    )
    
    # --- 设置初始参数（可选）---
    initial_params = None

    # --- 执行优化 ---
    print("\n" + "=" * 60)
    print("开始CMA-ES优化...")
    print("=" * 60)
    
    result = optimizer.optimize(
        target_mask=target_mask,
        initial_params=initial_params,
        sigma=CMAES_SIGMA,
        popsize=CMAES_POPSIZE,
        maxiter=CMAES_MAXITER
    )
    
    # --- 输出结果 ---
    print("\n" + "=" * 60)
    print("优化结果 (Z-Up 坐标系)")
    print("=" * 60)
    print(f"最优方位角: {result['azimuth']:.2f}°")
    print(f"最优仰角: {result['elevation']:.2f}°")
    print(f"最优距离: {result['distance']:.4f} 单位")
    print(f"相对距离: {result['distance'] / optimizer.mesh_scale:.2f}x (相对于网格对角线)")
    print(f"最终IoU: {result['iou']:.4f}")
    print(f"总迭代次数: {result['iterations']}")
    
    # --- 自检模式：比较结果 ---
    if run_self_check and gt_params is not None:
        print("\n" + "-" * 40)
        print("自检模式 - 参数对比")
        print("-" * 40)
        
        azimuth_error = np.abs(np.radians(result['azimuth']) - gt_params['azimuth'])
        # 处理方位角的周期性
        azimuth_error = min(azimuth_error, 2 * np.pi - azimuth_error)
        
        elevation_error = np.abs(result['elevation'] - gt_params['elevation'])
        distance_error = np.abs(result['distance'] - gt_params['distance'])
        
        print(f"方位角误差: {np.degrees(azimuth_error):.2f}°")
        print(f"仰角误差: {elevation_error:.2f}°")
        print(f"距离误差: {distance_error:.4f} 单位")
        
        if azimuth_error < np.radians(5) and elevation_error < 5 and distance_error < 0.2 * optimizer.mesh_scale:
            print("\n✓ 自检通过！估计参数与真实参数接近。")
        else:
            print("\n✗ 自检警告：估计参数与真实参数存在较大偏差。")
    
    # --- 生成可视化 ---
    if SAVE_VISUALIZATION:
        print("\n生成可视化结果...")
        
        # 使用内置的可视化方法
        viz_path = os.path.join(OUTPUT_DIR, 'optimization_result.png')
        optimizer.visualize_result(target_mask, result, save_path=viz_path)
        
        # 保存优化后的掩码（注意：result['azimuth']是度数，需要转换为弧度）
        optimized_mask = optimizer.render_silhouette(
            np.radians(result['azimuth']),
            result['elevation'],
            result['distance']
        )
        # 调整target_mask尺寸到render_resolution
        target_mask_resized = cv2.resize(target_mask, optimizer.render_resolution, interpolation=cv2.INTER_NEAREST)
        # 对齐渲染mask到target mask
        optimized_mask_aligned = optimizer.align_masks(optimized_mask, target_mask_resized)
        mask_output_path = os.path.join(OUTPUT_DIR, 'optimized_mask.png')
        cv2.imwrite(mask_output_path, optimized_mask_aligned)
        print(f"已保存优化掩码: {mask_output_path}")
        
        # 保存参数到文本文件
        params_output_path = os.path.join(OUTPUT_DIR, 'estimated_parameters.txt')
        with open(params_output_path, 'w', encoding='utf-8') as f:
            f.write("6D位姿估计结果\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"网格文件: {MESH_PATH}\n")
            f.write(f"目标掩码: {TARGET_MASK_PATH}\n\n")
            f.write("估计的相机参数 (Z-Up 坐标系):\n")
            f.write(f"  方位角 (Azimuth): {result['azimuth']:.4f}°\n")
            f.write(f"  仰角 (Elevation): {result['elevation']:.4f}°\n")
            f.write(f"  距离 (Distance): {result['distance']:.4f} 单位\n")
            f.write(f"  相对距离 (Relative): {result['distance'] / optimizer.mesh_scale:.4f}x 网格对角线\n\n")
            f.write(f"网格信息:\n")
            f.write(f"  包围盒中心: {optimizer.mesh_center}\n")
            f.write(f"  网格尺寸: {optimizer.mesh_scale:.4f}\n")
            f.write(f"  坐标系转换: Z-Up (原始) -> Y-Up (渲染)\n\n")
            f.write(f"优化指标:\n")
            f.write(f"  最终IoU: {result['iou']:.6f}\n")
            f.write(f"  迭代次数: {result['iterations']}\n\n")
            f.write("相机变换矩阵 (Y-Up 坐标系，用于 pyrender 渲染):\n")
            f.write(np.array2string(result['camera_pose'], precision=6, suppress_small=True))
        print(f"已保存参数文件: {params_output_path}")
    
    # --- 输出相机变换矩阵 ---
    print("\n相机变换矩阵 (Y-Up 坐标系，用于 pyrender):")
    print(result['camera_pose'])
    
    # --- 交互式可视化 ---
    if SHOW_INTERACTIVE:
        optimizer.show_interactive_result(result)
    
    # 清理资源
    optimizer.cleanup()
    
    print("\n" + "=" * 60)
    print("位姿估计完成!")
    print("=" * 60)
    
    return result, result['camera_pose']


if __name__ == "__main__":
    main()