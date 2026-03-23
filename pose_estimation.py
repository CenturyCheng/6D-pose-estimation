"""
基于法线匹配的6D位姿估计
=============================================
使用CMA-ES优化算法，通过将3D网格渲染的法线图与目标法线图对齐，
来估计相机参数（方位角、仰角、距离）。

渲染背景为黑色，通过AABB包围盒匹配对齐，使用RGB归一化向量差计算相似度。

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
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing

# ============================================================================
# 可调整参数配置区
# ============================================================================

# --- 文件路径配置 ---
MESH_PATH = "./model.glb"           # 3D网格文件路径（支持 .obj, .stl, .ply 等）
TARGET_MASK_PATH = "./target.png"   # 目标二值掩码图像路径
OUTPUT_DIR = "./output"             # 输出结果保存目录

# --- 渲染参数 ---
RENDER_WIDTH = 512                  # 渲染图像宽度（像素）
RENDER_HEIGHT = 512                 # 渲染图像高度（像素）
CAMERA_FOV = 45.0                   # 相机视场角（度）

# --- CMA-ES优化参数 ---
CMAES_SIGMA = 1.5                   # CMA-ES全局初始步长（各维度由CMA_stds控制）
CMAES_POPSIZE = 80                  # 种群大小 - 更大种群提高全局搜索覆盖度
CMAES_MAXITER = 100                 # 最大迭代次数
CMAES_TOLX = 1e-4                   # 参数变化容差（收敛条件）
CMAES_TOLFUN = 5e-5                 # 目标函数值容差（收敛条件）

# --- 并行计算参数 ---
USE_PARALLEL = True                 # 是否使用并行计算
NUM_WORKERS = None                  # 并行进程数（None=自动：CPU核数的一半，最少1个，最多留4核给系统）

# --- 参数搜索范围 ---
AZIMUTH_RANGE = (0.0, 2 * np.pi)    # 方位角范围（弧度），0~2π
ELEVATION_RANGE = (-60.0, 80.0)     # 仰角范围（度），限制为合理的正面视角（避免俯视）
DISTANCE_MIN = 1.0                  # 最小相对距离（相对于网格对角线的倍数）
DISTANCE_MAX = 20.0                 # 最大相对距离
# CMA-ES 在对数空间优化距离: actual_dist = mesh_scale * exp(log_rel_dist)
LOG_DIST_MIN = np.log(DISTANCE_MIN) 
LOG_DIST_MAX = np.log(DISTANCE_MAX) 

# --- 其他设置 ---
VERBOSE = True                      # 是否打印详细信息
SAVE_VISUALIZATION = True           # 是否保存可视化结果
SHOW_INTERACTIVE = False            # 是否显示交互式pyrender窗口
DEBUG_OUTPUT = True                  # 是否输出调试渲染图像

# ============================================================================
# 设置无头渲染环境（避免窗口闪烁）
# ============================================================================
# 注意：在Windows上，并行渲染时设置osmesa可以避免窗口闪烁
# 但需要确保在子进程中也设置此环境变量
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'


def pad_to_square(image: np.ndarray, pad_value: int = 0) -> np.ndarray:
    """
    将非正方形图像填充为正方形，保持宽高比不变。
    
    填充策略：以较长边为基准，在较短边的两侧均匀填充黑色像素。
    
    参数:
        image: 输入图像 (H, W) 或 (H, W, C)
        pad_value: 填充值，默认为0（黑色）
    
    返回:
        正方形图像，尺寸为 (max(H,W), max(H,W)) 或 (max(H,W), max(H,W), C)
    """
    h, w = image.shape[:2]
    
    # 如果已经是正方形，直接返回
    if h == w:
        return image
    
    # 计算目标尺寸（取较长边）
    size = max(h, w)
    
    # 计算填充量
    pad_h = size - h
    pad_w = size - w
    
    # 均匀分配到两侧
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # 根据图像维度进行填充
    if len(image.shape) == 2:
        # 灰度图
        padded = np.full((size, size), pad_value, dtype=image.dtype)
        padded[pad_top:pad_top+h, pad_left:pad_left+w] = image
    else:
        # 彩色图
        padded = np.full((size, size, image.shape[2]), pad_value, dtype=image.dtype)
        padded[pad_top:pad_top+h, pad_left:pad_left+w, :] = image
    
    return padded


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


# ============================================================================
# 并行计算辅助函数 - 使用全局变量缓存渲染器
# ============================================================================

# 全局变量：在每个worker进程中缓存渲染器资源
_worker_renderer = None
_worker_scene = None
_worker_mesh = None
_worker_camera_node = None

def _init_worker(mesh_path: str, render_resolution: Tuple[int, int]):
    """
    初始化worker进程的渲染资源（每个进程只调用一次）
    """
    global _worker_renderer, _worker_scene, _worker_mesh, _worker_camera_node

    import numpy as np
    import trimesh
    import pyrender
    import warnings

    # 抑制警告
    warnings.filterwarnings('ignore')

    # 加载网格（只加载一次）
    mesh = trimesh.load(mesh_path, force='mesh')
    _worker_mesh = mesh

    # 创建场景
    _worker_scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 1.0],
        ambient_light=[0.3, 0.3, 0.3]
    )

    # 添加网格
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0
    )
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    _worker_scene.add(pyrender_mesh)

    # 添加光源
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    _worker_scene.add(light, pose=np.eye(4))

    # 创建渲染器（只创建一次）
    _worker_renderer = pyrender.OffscreenRenderer(
        viewport_width=render_resolution[0],
        viewport_height=render_resolution[1]
    )

    _worker_camera_node = None


def _evaluate_single_solution(
    params: np.ndarray,
    target_normal: np.ndarray,
    mesh_center: np.ndarray,
    mesh_scale: float
) -> float:
    """
    在worker进程中评估单个解（使用缓存的渲染器）

    参数:
        params: [azimuth, elevation, log_rel_dist] 参数
        target_normal: 目标法线图
        mesh_center: 网格中心
        mesh_scale: 网格尺度

    返回:
        负相似度值（因为CMA-ES最小化）
    """
    global _worker_renderer, _worker_scene, _worker_mesh, _worker_camera_node

    import numpy as np
    import cv2

    # 规范化参数
    azimuth, elevation, log_rel_dist = params
    azimuth = azimuth % (2 * np.pi)
    elevation = np.clip(elevation, ELEVATION_RANGE[0], ELEVATION_RANGE[1])
    log_rel_dist = np.clip(log_rel_dist, LOG_DIST_MIN, LOG_DIST_MAX)
    distance = mesh_scale * np.exp(log_rel_dist)

    # 计算相机位置（球坐标转笛卡尔坐标）
    el = np.radians(elevation)
    y = distance * np.sin(el)
    r_horizontal = distance * np.cos(el)
    x = r_horizontal * np.cos(azimuth)
    z = r_horizontal * np.sin(azimuth)
    camera_position = np.array([x, y, z]) + mesh_center

    # 计算相机位姿
    eye = camera_position
    target = mesh_center
    up = np.array([0.0, 1.0, 0.0])

    z_axis = eye - target
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross(np.array([1.0, 0.0, 0.0]), z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    camera_pose = np.eye(4, dtype=np.float64)
    camera_pose[:3, 0] = x_axis
    camera_pose[:3, 1] = y_axis
    camera_pose[:3, 2] = z_axis
    camera_pose[:3, 3] = eye

    # 计算自适应FOV
    bounds = _worker_mesh.bounds
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

    to_center = mesh_center - camera_position
    dist_to_center = np.linalg.norm(to_center)
    right = x_axis
    up_vec = y_axis

    max_horizontal = 0.0
    max_vertical = 0.0
    for corner in corners:
        to_corner = corner - camera_position
        horizontal_proj = np.dot(to_corner, right)
        vertical_proj = np.dot(to_corner, up_vec)
        horizontal_angle = np.abs(np.arctan2(horizontal_proj, dist_to_center))
        vertical_angle = np.abs(np.arctan2(vertical_proj, dist_to_center))
        max_horizontal = max(max_horizontal, horizontal_angle)
        max_vertical = max(max_vertical, vertical_angle)

    fov_horizontal = 2 * max_horizontal * 1.15
    fov_vertical = 2 * max_vertical * 1.15

    # 获取渲染器的宽高比
    render_width = _worker_renderer.viewport_width
    render_height = _worker_renderer.viewport_height
    aspect_ratio = render_width / render_height

    fov_vertical_from_horizontal = 2 * np.arctan(np.tan(fov_horizontal / 2) / aspect_ratio)
    adaptive_fov = max(fov_vertical, fov_vertical_from_horizontal)
    adaptive_fov_deg = np.degrees(adaptive_fov)
    adaptive_fov_deg = np.clip(adaptive_fov_deg, 10.0, 120.0)

    # 更新场景中的相机
    import pyrender
    camera = pyrender.PerspectiveCamera(
        yfov=np.radians(adaptive_fov_deg),
        aspectRatio=aspect_ratio
    )

    # 移除旧相机，添加新相机
    if _worker_camera_node is not None:
        _worker_scene.remove_node(_worker_camera_node)
    _worker_camera_node = _worker_scene.add(camera, pose=camera_pose)

    # 使用缓存的渲染器渲染
    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    color, depth = _worker_renderer.render(_worker_scene, flags=flags)

    # 从深度图生成法线图
    h, w = depth.shape
    fx = fy = (h / 2.0) / np.tan(np.radians(adaptive_fov_deg) / 2.0)
    cx, cy = w / 2.0, h / 2.0

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x_img = (u - cx) * depth / fx
    y_img = (v - cy) * depth / fy
    z_img = depth

    x_padded = np.pad(x_img, ((1, 1), (1, 1)), mode='edge')
    y_padded = np.pad(y_img, ((1, 1), (1, 1)), mode='edge')
    z_padded = np.pad(z_img, ((1, 1), (1, 1)), mode='edge')

    tx = (x_padded[1:-1, 2:] - x_padded[1:-1, :-2]) / 2.0
    ty = (y_padded[1:-1, 2:] - y_padded[1:-1, :-2]) / 2.0
    tz = (z_padded[1:-1, 2:] - z_padded[1:-1, :-2]) / 2.0

    bx = (x_padded[2:, 1:-1] - x_padded[:-2, 1:-1]) / 2.0
    by = (y_padded[2:, 1:-1] - y_padded[:-2, 1:-1]) / 2.0
    bz = (z_padded[2:, 1:-1] - z_padded[:-2, 1:-1]) / 2.0

    normal_x = ty * bz - tz * by
    normal_y = tz * bx - tx * bz
    normal_z = tx * by - ty * bx

    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    norm = np.maximum(norm, 1e-8)

    normal_x /= norm
    normal_y /= norm
    normal_z /= norm

    flip_mask = normal_z < 0
    normal_x = np.where(flip_mask, -normal_x, normal_x)
    normal_y = np.where(flip_mask, -normal_y, normal_y)
    normal_z = np.where(flip_mask, -normal_z, normal_z)

    normal_x = -normal_x * 0.5 + 0.5
    normal_y = normal_y * 0.5 + 0.5
    normal_z = normal_z * 0.5 + 0.5

    normal_map = np.stack([normal_x, normal_y, normal_z], axis=-1)
    normal_map = (normal_map * 255).astype(np.uint8)

    mask = depth > 1e-3
    normal_map[~mask] = [0, 0, 0]

    rendered_normal = normal_map

    # 调整尺寸
    # 调整尺寸
    if rendered_normal.shape[:2] != target_normal.shape[:2]:
        rendered_normal = cv2.resize(
            rendered_normal,
            (target_normal.shape[1], target_normal.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    # 计算AABB并对齐
    def compute_bbox(image, is_target=False):
        if is_target:
            brightness = np.max(image, axis=-1)
            fg_mask = brightness >= 6
        else:
            fg_mask = np.any(image > 0, axis=-1)
        y_coords, x_coords = np.where(fg_mask)
        if len(x_coords) == 0:
            return (0, 0, image.shape[1], image.shape[0])
        return (int(np.min(x_coords)), int(np.min(y_coords)),
               int(np.max(x_coords)), int(np.max(y_coords)))

    r_bbox = compute_bbox(rendered_normal, is_target=False)
    t_bbox = compute_bbox(target_normal, is_target=True)

    r_cx = (r_bbox[0] + r_bbox[2]) / 2.0
    r_cy = (r_bbox[1] + r_bbox[3]) / 2.0
    r_w = r_bbox[2] - r_bbox[0]
    r_h = r_bbox[3] - r_bbox[1]

    t_cx = (t_bbox[0] + t_bbox[2]) / 2.0
    t_cy = (t_bbox[1] + t_bbox[3]) / 2.0
    t_w = t_bbox[2] - t_bbox[0]
    t_h = t_bbox[3] - t_bbox[1]

    if r_w > 0 and r_h > 0:
        scale_x = t_w / r_w
        scale_y = t_h / r_h
    else:
        scale_x = 1.0
        scale_y = 1.0

    M = np.array([
        [scale_x, 0, t_cx - scale_x * r_cx],
        [0, scale_y, t_cy - scale_y * r_cy]
    ], dtype=np.float32)

    aligned = cv2.warpAffine(
        rendered_normal,
        M,
        (target_normal.shape[1], target_normal.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # 计算相似度
    aligned_fg_full = np.any(aligned > 0, axis=-1)
    target_fg_full = np.max(target_normal, axis=-1) >= 6

    intersection = np.sum(aligned_fg_full & target_fg_full)
    union = np.sum(aligned_fg_full | target_fg_full)
    silhouette_iou = intersection / max(union, 1)

    min_x, min_y, max_x, max_y = t_bbox
    region_aligned = aligned[min_y:max_y+1, min_x:max_x+1].astype(np.float32)
    region_target = target_normal[min_y:max_y+1, min_x:max_x+1].astype(np.float32)

    aligned_fg = np.any(region_aligned > 0, axis=-1)
    target_fg = np.max(region_target, axis=-1) >= 6
    valid_mask = aligned_fg & target_fg

    valid_count = np.sum(valid_mask)
    if valid_count == 0:
        similarity = float(0.5 * silhouette_iou)
    else:
        v_aligned = region_aligned[valid_mask]
        v_target = region_target[valid_mask]

        norm_a = np.linalg.norm(v_aligned, axis=1, keepdims=True)
        norm_t = np.linalg.norm(v_target, axis=1, keepdims=True)
        norm_a = np.maximum(norm_a, 1e-8)
        norm_t = np.maximum(norm_t, 1e-8)

        v_aligned_unit = v_aligned / norm_a
        v_target_unit = v_target / norm_t

        diff = np.linalg.norm(v_aligned_unit - v_target_unit, axis=1)
        mean_diff = np.mean(diff)
        normal_similarity = 1.0 - mean_diff / 2.0

        similarity = 0.5 * normal_similarity + 0.5 * silhouette_iou

    # 返回负相似度（CMA-ES最小化）
    return -similarity


# ============================================================================
# 位姿优化器类
# ============================================================================

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
        # 创建场景 - 黑色背景
        self.scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 1.0],  # 黑色背景
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

            # 计算视角（使用到网格中心的距离）
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
        
        # 限制FOV在合理范围内
        # 下限不能太小，否则趋近正交投影，失去透视效果
        fov_deg = np.clip(fov_deg, 10.0, 120.0)
        
        return fov_deg
    
    def depth_to_view_space_normal(self, depth: np.ndarray, camera_pose: np.ndarray, fov: float = None) -> np.ndarray:
        """
        从深度图计算视图空间法线（切线空间法线）

        参数:
            depth: 深度图 (H, W)
            camera_pose: 相机位姿矩阵
            fov: 实际使用的FOV（度），如果为None则使用self.fov

        返回:
            视图空间法线图 (H, W, 3)，RGB值范围[0, 255]
        """
        h, w = depth.shape

        # 使用实际渲染时的FOV计算内参
        actual_fov = fov if fov is not None else self.fov
        fx = fy = (h / 2.0) / np.tan(np.radians(actual_fov) / 2.0)
        cx, cy = w / 2.0, h / 2.0

        # 重建3D点云（相机空间）
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        # 使用交叉乘积计算表面法线
        # 对于每个像素，使用其上下左右邻居计算切线向量，然后计算法线

        # 计算x和y方向的偏导数（使用中心差分）
        # dx = (右边点 - 左边点) / 2
        # dy = (下边点 - 上边点) / 2

        # Pad arrays to handle边界
        x_padded = np.pad(x, ((1, 1), (1, 1)), mode='edge')
        y_padded = np.pad(y, ((1, 1), (1, 1)), mode='edge')
        z_padded = np.pad(z, ((1, 1), (1, 1)), mode='edge')

        # 计算沿u方向的切向量 (右 - 左)
        tx = (x_padded[1:-1, 2:] - x_padded[1:-1, :-2]) / 2.0
        ty = (y_padded[1:-1, 2:] - y_padded[1:-1, :-2]) / 2.0
        tz = (z_padded[1:-1, 2:] - z_padded[1:-1, :-2]) / 2.0

        # 计算沿v方向的切向量 (下 - 上)
        bx = (x_padded[2:, 1:-1] - x_padded[:-2, 1:-1]) / 2.0
        by = (y_padded[2:, 1:-1] - y_padded[:-2, 1:-1]) / 2.0
        bz = (z_padded[2:, 1:-1] - z_padded[:-2, 1:-1]) / 2.0

        # 法线 = 切向量的叉乘 (tangent_u × tangent_v)
        # 注意OpenGL/相机坐标系：X右，Y上，Z向后（指向相机）
        # 所以需要反转法线方向使其指向相机
        normal_x = ty * bz - tz * by
        normal_y = tz * bx - tx * bz
        normal_z = tx * by - ty * bx

        # 归一化法线
        norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        norm = np.maximum(norm, 1e-8)  # 避免除零

        normal_x /= norm
        normal_y /= norm
        normal_z /= norm

        # 确保法线指向相机（Z分量应该是正的，因为相机看向-Z）
        # 如果Z是负的，翻转法线
        flip_mask = normal_z < 0
        normal_x = np.where(flip_mask, -normal_x, normal_x)
        normal_y = np.where(flip_mask, -normal_y, normal_y)
        normal_z = np.where(flip_mask, -normal_z, normal_z)

        # 转换到 [0, 1] 范围
        # 法线从 [-1, 1] 映射到 [0, 1]，然后到 [0, 255]
        # 注意：翻转X轴以匹配Sunshineflow的法线约定（右侧=红色）
        normal_x = -normal_x * 0.5 + 0.5  # 翻转X
        normal_y = normal_y * 0.5 + 0.5
        normal_z = normal_z * 0.5 + 0.5

        # 组合成RGB图像 (R=X, G=Y, B=Z)
        normal_map = np.stack([normal_x, normal_y, normal_z], axis=-1)

        # 转换到 [0, 255] 范围
        normal_map = (normal_map * 255).astype(np.uint8)

        # 对于背景区域（depth=0或非常小），填充黑色 (0, 0, 0)
        mask = depth > 1e-3
        normal_map[~mask] = [0, 0, 0]

        return normal_map

    def render_silhouette(
        self,
        azimuth: float,
        elevation: float,
        distance: float,
        use_adaptive_fov: bool = True,
        return_depth: bool = False,
        return_normal: bool = False
    ) -> np.ndarray:
        """
        渲染给定相机参数下的网格轮廓、深度图或法线图。

        参数:
            azimuth: 方位角（弧度）
            elevation: 仰角（度）
            distance: 距离
            use_adaptive_fov: 是否使用自适应FOV（默认True）
            return_depth: 是否返回深度图（默认False）
            return_normal: 是否返回法线图（默认False）

        返回:
            如果return_normal=True: 法线图 (H, W, 3)，RGB值范围[0, 255]
            如果return_depth=True: 深度图 (H, W)，值为深度值
            否则: 二值轮廓图像 (H, W)，值为0或255
        """
        # 获取相机位姿
        camera_pose = self.get_camera_pose(azimuth, elevation, distance)

        # 计算自适应FOV
        if use_adaptive_fov:
            adaptive_fov = self.compute_adaptive_fov(azimuth, elevation, distance, margin=0.15)
            actual_fov = adaptive_fov
            # 创建新的相机实例，避免修改全局相机FOV
            camera = pyrender.PerspectiveCamera(
                yfov=np.radians(adaptive_fov),
                aspectRatio=self.render_resolution[0] / self.render_resolution[1]
            )
        else:
            actual_fov = self.fov
            camera = self.camera

        # 更新场景中的相机
        if self.camera_node is not None:
            self.scene.remove_node(self.camera_node)
        self.camera_node = self.scene.add(camera, pose=camera_pose)

        # 渲染
        if return_normal:
            # 渲染法线图（使用pyrender的normals标志）
            flags = pyrender.RenderFlags.SKIP_CULL_FACES
            color, depth = self.renderer.render(self.scene, flags=flags)

            # pyrender返回的color已经包含了法线信息（当材质设置正确时）
            # 但我们需要直接获取几何法线，所以重新渲染获取法线
            # 使用自定义shader或通过深度重建法线
            normal_map = self.depth_to_view_space_normal(depth, camera_pose, fov=actual_fov)
            return normal_map
        else:
            color, depth = self.renderer.render(self.scene)

        if return_depth:
            # 返回深度图
            return depth
        else:
            # 从深度图生成轮廓（深度>0的像素为前景）
            silhouette = (depth > 6).astype(np.uint8) * 255
            return silhouette
    
    def compute_foreground_bbox(self, image: np.ndarray, is_target: bool = False) -> Tuple[int, int, int, int]:
        """
        计算图像前景区域的AABB包围盒。

        对于渲染图：非黑色区域（RGB任一通道 > 0）
        对于目标图：明度 >= 6 的区域

        参数:
            image: 输入图像 (H, W, 3) RGB格式
            is_target: 是否为目标图（使用明度判断）

        返回:
            (min_x, min_y, max_x, max_y) 包围盒坐标
        """
        if is_target:
            # 目标图：计算明度，明度 >= 6 的区域为主体
            # 明度 = max(R, G, B)
            brightness = np.max(image, axis=-1)
            fg_mask = brightness >= 6
        else:
            # 渲染图：非黑色区域（任一通道 > 0）
            fg_mask = np.any(image > 0, axis=-1)

        y_coords, x_coords = np.where(fg_mask)

        if len(x_coords) == 0:
            # 如果前景为空，返回整个图像
            return (0, 0, image.shape[1], image.shape[0])

        min_x = int(np.min(x_coords))
        max_x = int(np.max(x_coords))
        min_y = int(np.min(y_coords))
        max_y = int(np.max(y_coords))

        return (min_x, min_y, max_x, max_y)
    
    def align_rendered_to_target(
        self,
        rendered_normal: np.ndarray,
        target_normal: np.ndarray
    ) -> np.ndarray:
        """
        将渲染法线图通过AABB box匹配对齐到目标法线图。

        步骤：
        1. 计算渲染图非黑色区域的AABB
        2. 计算目标图明度>=6区域的AABB
        3. 将渲染图的AABB平移缩放到目标图的AABB位置

        参数:
            rendered_normal: 渲染的法线图 (H, W, 3) RGB
            target_normal: 目标法线图 (H, W, 3) RGB

        返回:
            对齐后的渲染法线图 (H, W, 3)，背景为黑色
        """
        # 计算两个图像的AABB
        r_bbox = self.compute_foreground_bbox(rendered_normal, is_target=False)
        t_bbox = self.compute_foreground_bbox(target_normal, is_target=True)

        # 渲染图AABB的中心和尺寸
        r_cx = (r_bbox[0] + r_bbox[2]) / 2.0
        r_cy = (r_bbox[1] + r_bbox[3]) / 2.0
        r_w = r_bbox[2] - r_bbox[0]
        r_h = r_bbox[3] - r_bbox[1]

        # 目标图AABB的中心和尺寸
        t_cx = (t_bbox[0] + t_bbox[2]) / 2.0
        t_cy = (t_bbox[1] + t_bbox[3]) / 2.0
        t_w = t_bbox[2] - t_bbox[0]
        t_h = t_bbox[3] - t_bbox[1]

        # 计算独立的X/Y缩放比例
        # 使用独立缩放而非统一缩放，消除透视造成的宽高比差异对匹配的影响
        if r_w > 0 and r_h > 0:
            scale_x = t_w / r_w
            scale_y = t_h / r_h
        else:
            scale_x = 1.0
            scale_y = 1.0

        scale = (scale_x + scale_y) / 2.0  # 用于返回值的平均缩放

        # 创建仿射变换矩阵：独立X/Y缩放后平移
        # new_x = scale_x * old_x + (t_cx - scale_x * r_cx)
        # new_y = scale_y * old_y + (t_cy - scale_y * r_cy)
        M = np.array([
            [scale_x, 0, t_cx - scale_x * r_cx],
            [0, scale_y, t_cy - scale_y * r_cy]
        ], dtype=np.float32)

        # 对法线图应用变换，背景填充黑色
        aligned = cv2.warpAffine(
            rendered_normal,
            M,
            (target_normal.shape[1], target_normal.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        return aligned, scale

    def compute_normal_similarity(
        self,
        rendered_normal: np.ndarray,
        target_normal: np.ndarray,
        align: bool = True
    ) -> float:
        """
        计算两个法线图之间的相似度（综合法线方向 + 轮廓形状）。

        流程：
        1. 将渲染法线图通过AABB对齐到目标法线图
        2. 在对齐后的全图上计算轮廓IoU（前景掩码的交集/并集）
        3. 在目标AABB区域内，对重叠像素做RGB归一化向量差
        4. 综合评分 = 0.5 * normal_sim + 0.5 * silhouette_iou

        参数:
            rendered_normal: 渲染的法线图 (H, W, 3) RGB
            target_normal: 目标法线图 (H, W, 3) RGB
            align: 是否在计算前对齐

        返回:
            相似度值 [0, 1]，1表示完全相同
        """
        if align:
            aligned, scale = self.align_rendered_to_target(rendered_normal, target_normal)
        else:
            aligned = rendered_normal.copy()
            scale = 1.0

        # === 1. 轮廓IoU（全图） ===
        # 对齐后的渲染图前景掩码
        aligned_fg_full = np.any(aligned > 0, axis=-1)
        # 目标图前景掩码
        target_fg_full = np.max(target_normal, axis=-1) >= 6

        intersection = np.sum(aligned_fg_full & target_fg_full)
        union = np.sum(aligned_fg_full | target_fg_full)
        silhouette_iou = intersection / max(union, 1)

        # === 2. 法线方向相似度（AABB区域内） ===
        t_bbox = self.compute_foreground_bbox(target_normal, is_target=True)
        min_x, min_y, max_x, max_y = t_bbox

        # 裁剪AABB区域
        region_aligned = aligned[min_y:max_y+1, min_x:max_x+1].astype(np.float32)
        region_target = target_normal[min_y:max_y+1, min_x:max_x+1].astype(np.float32)

        # 在AABB区域内，找到两张图都有前景的像素
        aligned_fg = np.any(region_aligned > 0, axis=-1)
        target_fg = np.max(region_target, axis=-1) >= 6
        valid_mask = aligned_fg & target_fg

        valid_count = np.sum(valid_mask)
        if valid_count == 0:
            # 没有重叠像素，只靠轮廓IoU
            return float(0.5 * silhouette_iou)

        # 提取有效像素
        v_aligned = region_aligned[valid_mask]  # (N, 3)
        v_target = region_target[valid_mask]    # (N, 3)

        # RGB归一化为单位向量
        norm_a = np.linalg.norm(v_aligned, axis=1, keepdims=True)
        norm_t = np.linalg.norm(v_target, axis=1, keepdims=True)
        norm_a = np.maximum(norm_a, 1e-8)
        norm_t = np.maximum(norm_t, 1e-8)

        v_aligned_unit = v_aligned / norm_a
        v_target_unit = v_target / norm_t

        # 计算归一化向量差的L2距离
        diff = np.linalg.norm(v_aligned_unit - v_target_unit, axis=1)

        # 归一化到 [0, 1]：diff 范围 [0, 2]，所以 similarity = 1 - mean(diff)/2
        mean_diff = np.mean(diff)
        normal_similarity = 1.0 - mean_diff / 2.0

        # === 3. 综合评分 ===
        # 50% 法线方向匹配 + 50% 轮廓形状匹配
        # 轮廓IoU直接惩罚形状不匹配（多出或缺少的区域）
        # 法线相似度衡量重叠区域的颜色/方向匹配质量
        final_similarity = 0.5 * normal_similarity + 0.5 * silhouette_iou

        return float(final_similarity)

    def _normalize_params(self, params: np.ndarray) -> Tuple[float, float, float]:
        """
        规范化参数：处理方位角周期性，限制仰角和距离范围。

        参数:
            params: [azimuth, elevation, log_rel_dist] 原始参数
                    其中 log_rel_dist 是对数相对距离

        返回:
            (azimuth, elevation, distance) 规范化后的参数（distance为绝对距离）
        """
        azimuth, elevation, log_rel_dist = params

        # 方位角：周期性处理（0到2π）
        azimuth = azimuth % (2 * np.pi)

        # 仰角：限制范围，避免万向锁
        elevation = np.clip(elevation, ELEVATION_RANGE[0], ELEVATION_RANGE[1])

        # 距离：对数空间 -> 实际距离
        # log_rel_dist 限制在 [LOG_DIST_MIN, LOG_DIST_MAX]
        log_rel_dist = np.clip(log_rel_dist, LOG_DIST_MIN, LOG_DIST_MAX)
        distance = self.mesh_scale * np.exp(log_rel_dist)

        return azimuth, elevation, distance
    
    def _objective_function(self, params: np.ndarray, target_normal: np.ndarray) -> float:
        """
        CMA-ES的目标函数：返回负相似度（因为CMA-ES最小化目标函数）。

        参数:
            params: [azimuth, elevation, distance]
            target_normal: 目标法线图 (H, W, 3) RGB

        返回:
            负相似度值
        """
        # 规范化参数
        azimuth, elevation, distance = self._normalize_params(params)

        # 渲染法线图（使用动态FOV使主体占满画面，距离只影响透视程度）
        rendered_normal = self.render_silhouette(azimuth, elevation, distance,
                                                 use_adaptive_fov=True, return_normal=True)

        # 调整尺寸以匹配目标法线图
        if rendered_normal.shape[:2] != target_normal.shape[:2]:
            rendered_normal = cv2.resize(
                rendered_normal,
                (target_normal.shape[1], target_normal.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        # 计算法线相似度（自动对齐）
        similarity = self.compute_normal_similarity(rendered_normal, target_normal, align=True)

        # 调试输出
        DEBUG_RENDER_PROB = 0.02
        if np.random.random() < DEBUG_RENDER_PROB and DEBUG_OUTPUT:
            debug_dir = os.path.join(OUTPUT_DIR, 'debug_renders')
            os.makedirs(debug_dir, exist_ok=True)

            aligned, _ = self.align_rendered_to_target(rendered_normal, target_normal)

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            axes[0].imshow(target_normal)
            axes[0].set_title('Target Normal')
            axes[0].axis('off')

            axes[1].imshow(aligned)
            axes[1].set_title(f'Rendered (Aligned)\nAz:{np.degrees(azimuth):.1f} El:{elevation:.1f} D:{distance/self.mesh_scale:.2f}x')
            axes[1].axis('off')

            overlay = (target_normal.astype(np.float32) * 0.5 + aligned.astype(np.float32) * 0.5).astype(np.uint8)
            axes[2].imshow(overlay)
            axes[2].set_title(f'Overlay\nSimilarity: {similarity:.4f}')
            axes[2].axis('off')

            diff = np.linalg.norm(target_normal.astype(np.float32) - aligned.astype(np.float32), axis=2)
            axes[3].imshow(diff, cmap='cool', vmin=0, vmax=255*np.sqrt(3))
            axes[3].set_title('L2 Difference')
            axes[3].axis('off')

            plt.suptitle(f'Iteration {self.iteration_count}', fontsize=12)
            plt.tight_layout()

            save_path = os.path.join(debug_dir, f'iter_{self.iteration_count:05d}_sim_{similarity:.4f}.png')
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()

        # 记录最佳结果
        if similarity > self.best_iou:
            self.best_iou = similarity
            self.best_params = (azimuth, elevation, distance)

        self.iteration_count += 1

        # 记录历史
        self.history.append({
            'iteration': self.iteration_count,
            'iou': similarity,
            'azimuth': azimuth,
            'elevation': elevation,
            'distance': distance
        })

        # 定期打印进度
        if self.verbose and self.iteration_count % 20 == 0:
            print(f"  迭代 {self.iteration_count}: 相似度 = {similarity:.4f}, "
                  f"最佳 = {self.best_iou:.4f}")

        # 返回负相似度（因为CMA-ES最小化目标函数）
        return -similarity
    
    def optimize(
        self,
        target_mask: np.ndarray,
        initial_params: Optional[np.ndarray] = None,
        sigma: float = CMAES_SIGMA,
        popsize: int = CMAES_POPSIZE,
        maxiter: int = CMAES_MAXITER,
        use_parallel: bool = USE_PARALLEL,
        num_workers: Optional[int] = NUM_WORKERS
    ) -> Dict[str, Any]:
        """
        执行CMA-ES优化，寻找最佳相机参数。

        参数:
            target_mask: 目标二值掩码图像
            initial_params: 初始参数 [azimuth, elevation, distance]，None则随机初始化
            sigma: CMA-ES初始步长
            popsize: 种群大小
            maxiter: 最大迭代次数
            use_parallel: 是否使用并行计算
            num_workers: 并行进程数（None=自动检测）

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

        # 预处理目标法线图
        if len(target_mask.shape) == 2:
            target_mask = cv2.cvtColor(target_mask, cv2.COLOR_GRAY2RGB)
        elif len(target_mask.shape) == 3 and target_mask.shape[2] == 4:
            target_mask = cv2.cvtColor(target_mask, cv2.COLOR_BGRA2RGB)

        # 调整大小到渲染分辨率
        target_mask = cv2.resize(
            target_mask,
            self.render_resolution,
            interpolation=cv2.INTER_LINEAR
        )
        
        if self.verbose:
            print(f"\n开始CMA-ES优化...")
            print(f"  种群大小: {popsize}")
            print(f"  最大迭代: {maxiter}")
            print(f"  初始步长: {sigma}")
            if use_parallel:
                if num_workers is None:
                    # 自动模式：CPU核数的一半，最小1，最大不超过(CPU总数-4)
                    cpu_count = multiprocessing.cpu_count()
                    half_cpus = cpu_count // 2
                    max_workers = cpu_count - 4
                    actual_workers = max(1, min(half_cpus, max_workers))
                else:
                    actual_workers = num_workers
                print(f"  并行模式: 启用 ({actual_workers} 个进程)")
            else:
                print(f"  并行模式: 禁用 (串行评估)")
        
        # 初始参数
        if initial_params is None:
            # 从正面略微仰视的角度开始
            initial_params = np.array([
                0.0,                                        # azimuth: 正面（0°）
                20.0,                                       # elevation: 略微仰视（20°）
                np.log(2.5)                                 # log_rel_dist: log(2.5) ≈ 0.916
            ])
        
        if self.verbose:
            print(f"  初始参数: 方位角={np.degrees(initial_params[0]):.1f}°, "
                  f"仰角={initial_params[1]:.1f}°, "
                  f"相对距离={np.exp(initial_params[2]):.2f}x (log={initial_params[2]:.3f})")
        
        # CMA-ES优化选项
        # 各维度使用不同的初始标准差：
        #   方位角: ~1.5 rad (~86°) 覆盖大范围角度, 仰角: ~25° 覆盖[-30,60], log距离: ~0.4
        opts = {
            'popsize': popsize,
            'maxiter': maxiter,
            'tolx': CMAES_TOLX,
            'tolfun': CMAES_TOLFUN,
            'verb_disp': 0,  # 禁用CMA-ES的内置输出
            'CMA_stds': [1.5, 25.0, 0.4],  # 各维度初始标准差比例 - 更大以增加探索范围
            'bounds': [
                [0, ELEVATION_RANGE[0], LOG_DIST_MIN],      # 下界
                [2 * np.pi, ELEVATION_RANGE[1], LOG_DIST_MAX]  # 上界
            ]
        }
        
        # 执行CMA-ES优化
        es = cma.CMAEvolutionStrategy(initial_params, sigma, opts)

        # 如果使用并行计算，创建进程池
        if use_parallel:
            if num_workers is None:
                # 自动模式：CPU核数的一半，最小1，最大不超过(CPU总数-4)
                cpu_count = multiprocessing.cpu_count()
                half_cpus = cpu_count // 2
                max_workers = cpu_count - 4
                actual_workers = max(1, min(half_cpus, max_workers))
            else:
                actual_workers = num_workers

            # 创建带initializer的进程池，每个worker进程初始化一次渲染器
            executor = ProcessPoolExecutor(
                max_workers=actual_workers,
                initializer=_init_worker,
                initargs=(MESH_PATH, self.render_resolution)
            )

            # 创建部分应用的评估函数（不需要传递mesh_path和render_resolution）
            eval_func = partial(
                _evaluate_single_solution,
                target_normal=target_mask,
                mesh_center=self.mesh_center,
                mesh_scale=self.mesh_scale
            )
        else:
            executor = None

        try:
            while not es.stop():
                # 获取候选解
                solutions = es.ask()

                # 评估每个候选解
                if use_parallel and executor is not None:
                    # 并行评估
                    fitnesses = list(executor.map(eval_func, solutions))

                    # 更新历史记录（串行模式下在_objective_function中更新）
                    for sol, fitness in zip(solutions, fitnesses):
                        similarity = -fitness
                        azimuth, elevation, log_rel_dist = sol
                        azimuth = azimuth % (2 * np.pi)
                        elevation = np.clip(elevation, ELEVATION_RANGE[0], ELEVATION_RANGE[1])
                        log_rel_dist = np.clip(log_rel_dist, LOG_DIST_MIN, LOG_DIST_MAX)
                        distance = self.mesh_scale * np.exp(log_rel_dist)

                        self.iteration_count += 1
                        self.history.append({
                            'iteration': self.iteration_count,
                            'iou': similarity,
                            'azimuth': azimuth,
                            'elevation': elevation,
                            'distance': distance
                        })

                        if similarity > self.best_iou:
                            self.best_iou = similarity
                            self.best_params = (azimuth, elevation, distance)

                    # 定期打印进度
                    if self.verbose and self.iteration_count % (popsize * 5) == 0:
                        print(f"  迭代 {self.iteration_count // popsize}: "
                              f"当前最佳相似度 = {self.best_iou:.4f}")
                else:
                    # 串行评估（使用原有的_objective_function）
                    fitnesses = [
                        self._objective_function(sol, target_mask)
                        for sol in solutions
                    ]

                # 更新CMA-ES
                es.tell(solutions, fitnesses)

        finally:
            # 关闭进程池
            if executor is not None:
                executor.shutdown(wait=True)
        
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
        target_normal: np.ndarray,
        result: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """
        可视化优化结果。

        参数:
            target_normal: 目标法线图 (H, W, 3) RGB
            result: optimize()返回的结果字典
            save_path: 保存路径，None则显示
        """
        # 渲染最终结果
        rendered_normal = self.render_silhouette(
            np.radians(result['azimuth']),
            result['elevation'],
            result['distance'],
            use_adaptive_fov=True,
            return_normal=True
        )

        # 确保target是RGB格式
        if len(target_normal.shape) == 2:
            target_normal = cv2.cvtColor(target_normal, cv2.COLOR_GRAY2RGB)
        elif len(target_normal.shape) == 3 and target_normal.shape[2] == 4:
            target_normal = cv2.cvtColor(target_normal, cv2.COLOR_BGRA2RGB)

        # 调整尺寸
        target_normal = cv2.resize(target_normal, self.render_resolution, interpolation=cv2.INTER_LINEAR)

        # 对齐渲染法线图
        if rendered_normal.shape[:2] != target_normal.shape[:2]:
            rendered_normal = cv2.resize(rendered_normal, (target_normal.shape[1], target_normal.shape[0]),
                                        interpolation=cv2.INTER_LINEAR)
        aligned, _ = self.align_rendered_to_target(rendered_normal, target_normal)

        # 计算相似度
        similarity = self.compute_normal_similarity(rendered_normal, target_normal, align=True)

        # 创建对比图
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # 目标法线图
        axes[0].imshow(target_normal)
        axes[0].set_title('Target Normal Map')
        axes[0].axis('off')

        # 对齐后的渲染法线图
        axes[1].imshow(aligned)
        axes[1].set_title(f'Rendered Normal (Aligned)\nAz: {result["azimuth_deg"]:.1f} El: {result["elevation"]:.1f}\n'
                         f'D: {result["distance"]/self.mesh_scale:.2f}x')
        axes[1].axis('off')

        # 差异热图
        diff_map = np.linalg.norm(target_normal.astype(np.float32) - aligned.astype(np.float32), axis=2)
        axes[2].imshow(diff_map, cmap='cool', vmin=0, vmax=255*np.sqrt(3))
        axes[2].set_title(f'Pixel Difference\nSimilarity: {similarity:.4f}')
        axes[2].axis('off')

        # 优化历史
        if result['history']:
            similarities = [h['iou'] for h in result['history']]
            axes[3].plot(similarities, 'b-', alpha=0.3)
            window = min(20, len(similarities))
            if window > 1:
                smoothed = np.convolve(similarities, np.ones(window)/window, mode='valid')
                axes[3].plot(range(window-1, len(similarities)), smoothed, 'r-', linewidth=2)
            axes[3].set_xlabel('Evaluations')
            axes[3].set_ylabel('Normal Similarity')
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


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数：执行完整的法线匹配位姿估计流程。
    """
    print("=" * 60)
    print("基于法线匹配的6D位姿估计")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 加载目标法线图 ---
    target_normal_path = "./target_normal.png"
    print(f"\n加载目标法线图: {target_normal_path}")
    target_normal = cv2.imread(target_normal_path, cv2.IMREAD_COLOR)

    if target_normal is None:
        print(f"[错误] 无法加载目标法线图: {target_normal_path}")
        return None, None

    # OpenCV读取的是BGR格式，转换为RGB
    target_normal = cv2.cvtColor(target_normal, cv2.COLOR_BGR2RGB)

    # 检查并处理非正方形图像
    original_shape = target_normal.shape
    if original_shape[0] != original_shape[1]:
        print(f"检测到非正方形图像: {original_shape[1]}x{original_shape[0]}")
        target_normal = pad_to_square(target_normal, pad_value=0)
        print(f"已填充为正方形: {target_normal.shape[1]}x{target_normal.shape[0]}")

    print(f"目标法线图尺寸: {target_normal.shape}")
    # 统计非黑色前景像素
    fg_pixels = np.sum(np.max(target_normal, axis=-1) >= 6)
    print(f"目标法线图前景像素: {fg_pixels}")

    # --- 创建优化器 ---
    print(f"\n加载网格: {MESH_PATH}")
    optimizer = PoseOptimizer(
        mesh_path=MESH_PATH,
        render_resolution=(RENDER_WIDTH, RENDER_HEIGHT),
        fov=CAMERA_FOV,
        verbose=VERBOSE
    )

    # --- 执行优化 ---
    print("\n" + "=" * 60)
    print("开始CMA-ES优化...")
    print("=" * 60)

    result = optimizer.optimize(
        target_mask=target_normal,
        initial_params=None,
        sigma=CMAES_SIGMA,
        popsize=CMAES_POPSIZE,
        maxiter=CMAES_MAXITER
    )

    # --- 输出结果 ---
    print("\n" + "=" * 60)
    print("优化结果 (Z-Up 坐标系)")
    print("=" * 60)
    print(f"最优方位角: {result['azimuth']:.2f}")
    print(f"最优仰角: {result['elevation']:.2f}")
    print(f"最优距离: {result['distance']:.4f} 单位")
    print(f"相对距离: {result['distance'] / optimizer.mesh_scale:.2f}x (相对于网格对角线)")
    print(f"最终相似度: {result['iou']:.4f}")
    print(f"总迭代次数: {result['iterations']}")

    # --- 生成可视化 ---
    if SAVE_VISUALIZATION:
        print("\n生成可视化结果...")

        viz_path = os.path.join(OUTPUT_DIR, 'optimization_result.png')
        optimizer.visualize_result(target_normal, result, save_path=viz_path)

        # 保存优化后的法线图
        optimized_normal = optimizer.render_silhouette(
            np.radians(result['azimuth']),
            result['elevation'],
            result['distance'],
            use_adaptive_fov=True,
            return_normal=True
        )
        # 调整target尺寸到render_resolution
        target_resized = cv2.resize(target_normal, optimizer.render_resolution, interpolation=cv2.INTER_LINEAR)

        # 对齐渲染法线图到target
        optimized_aligned, _ = optimizer.align_rendered_to_target(optimized_normal, target_resized)

        # 保存法线图（RGB -> BGR for cv2）
        normal_output_path = os.path.join(OUTPUT_DIR, 'optimized_normal.png')
        normal_bgr = cv2.cvtColor(optimized_aligned, cv2.COLOR_RGB2BGR)
        cv2.imwrite(normal_output_path, normal_bgr)
        print(f"已保存优化法线图: {normal_output_path}")

        # 保存参数到文本文件
        params_output_path = os.path.join(OUTPUT_DIR, 'estimated_parameters.txt')
        with open(params_output_path, 'w', encoding='utf-8') as f:
            f.write("6D位姿估计结果（法线匹配）\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"网格文件: {MESH_PATH}\n")
            f.write(f"目标法线图: {target_normal_path}\n\n")
            f.write("估计的相机参数 (Z-Up 坐标系):\n")
            f.write(f"  方位角 (Azimuth): {result['azimuth']:.4f}\n")
            f.write(f"  仰角 (Elevation): {result['elevation']:.4f}\n")
            f.write(f"  距离 (Distance): {result['distance']:.4f} 单位\n")
            f.write(f"  相对距离 (Relative): {result['distance'] / optimizer.mesh_scale:.4f}x 网格对角线\n\n")
            f.write(f"网格信息:\n")
            f.write(f"  包围盒中心: {optimizer.mesh_center}\n")
            f.write(f"  网格尺寸: {optimizer.mesh_scale:.4f}\n\n")
            f.write(f"优化指标:\n")
            f.write(f"  最终相似度: {result['iou']:.6f}\n")
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