import os
import json
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ===================== 配置参数 =====================
GRID_SIZE = 20          # 20×20×20网格
TIME_STEPS = 10         # 10个时间步
SIGNAL_THRESHOLD = 0.2  # 通信阈值
SAFE_RADIUS = 2         # 避障安全半径（膨胀半径）

# ===================== 1. 静态层（硬约束） =====================
def generate_static_terrain(grid_size):
    terrain = np.random.choice([0,1], (grid_size, grid_size, grid_size), p=[0.99, 0.01])
    expanded_terrain = terrain.copy()
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                if terrain[x,y,z] == 1:
                    for dx in range(-SAFE_RADIUS, SAFE_RADIUS + 1):
                        for dy in range(-SAFE_RADIUS, SAFE_RADIUS + 1):
                            for dz in range(-SAFE_RADIUS, SAFE_RADIUS + 1):
                                nx, ny, nz = x+dx, y+dy, z+dz
                                if 0<=nx<grid_size and 0<=ny<grid_size and 0<=nz<grid_size:
                                    expanded_terrain[nx,ny,nz] = 1
    return expanded_terrain

# ===================== 2. 动态层（软约束 & 硬约束） =====================
@jit(nopython=True)
def generate_dynamic_current(grid_size, t):
    u = np.zeros((grid_size, grid_size, grid_size))
    v = np.zeros((grid_size, grid_size, grid_size))
    w = np.zeros((grid_size, grid_size, grid_size))
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                u[x,y,z] = 1.5 * np.sin(0.2*x + 0.1*t)  
                v[x,y,z] = 1.5 * np.cos(0.2*y + 0.1*t)  
                w[x,y,z] = 0.3 * np.sin(0.1*z + 0.05*t) 
    return u, v, w

def generate_dynamic_obstacles(grid_size, t, num_obstacles=12):
    obstacles =[]
    rng = np.random.RandomState(t * 10) 
    for _ in range(num_obstacles):
        x = (rng.randint(SAFE_RADIUS, grid_size-SAFE_RADIUS) + int(t*0.8)) % grid_size
        y = (rng.randint(SAFE_RADIUS, grid_size-SAFE_RADIUS) + int(t*0.5)) % grid_size
        z = rng.randint(SAFE_RADIUS, grid_size-SAFE_RADIUS)
        obstacles.append([int(x), int(y), int(z)])
    return obstacles

# ===================== 3. 约束掩码层 =====================
@jit(nopython=True)
def generate_signal_field(grid_size, comm_point=(5,5,2), threshold=SIGNAL_THRESHOLD):
    signal = np.zeros((grid_size, grid_size, grid_size))
    signal_mask = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    cx, cy, cz = comm_point
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                dist = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
                val = np.exp(-0.08 * dist) * (1 - 0.02*z)
                signal[x,y,z] = max(val, 0.0)
                if signal[x,y,z] < threshold:
                    signal_mask[x,y,z] = 100.0
                else:
                    signal_mask[x,y,z] = 0.0
    return signal, signal_mask

# ===================== 4. 整合时空地图 =====================
def generate_ocean_map():
    static_terrain = generate_static_terrain(GRID_SIZE)
    dynamic_map =[]
    for t in range(TIME_STEPS):
        cur_u, cur_v, cur_w = generate_dynamic_current(GRID_SIZE, t)
        obstacles = generate_dynamic_obstacles(GRID_SIZE, t)
        signal, signal_mask = generate_signal_field(GRID_SIZE)
        step_map = {
            "time_step": t, "static_terrain": static_terrain,
            "current_u": cur_u, "current_v": cur_v, "current_w": cur_w,
            "dynamic_obstacles": obstacles,
            "signal_strength": signal, "signal_mask": signal_mask,
        }
        dynamic_map.append(step_map)
    return dynamic_map

# ===================== 5. [新增] 绘制 2.5D 高清静态结构图 =====================
def generate_2_5d_snapshot(dynamic_map, t_index=0, output_path="snapshot.png"):
    print(f"📸 正在生成 T={t_index} 时刻的 2.5D 综合观测图...")
    step_data = dynamic_map[t_index]
    terrain = step_data["static_terrain"]
    grid_size = terrain.shape[0]
    
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"2.5D Ocean Environment Snapshot (Time Step: {t_index})", fontsize=16, pad=20)
    
    # --- A. 绘制静态地形 (Voxels) ---
    voxel_mask = terrain == 1
    colors = np.empty(terrain.shape, dtype=object)
    colors[voxel_mask] = '#4682B455'  # 钢蓝色 + 强透明度，防止遮挡内部
    ax.voxels(voxel_mask, facecolors=colors, edgecolors='#333333', linewidth=0.2)
    
    # --- B. 绘制洋流矢量场 (Quiver 箭头) ---
    # 抽样参数：每隔 4 个网格画一个箭头，避免变成刺猬
    step = 4
    x, y, z = np.meshgrid(np.arange(0, grid_size, step),
                          np.arange(0, grid_size, step),
                          np.arange(0, grid_size, step), indexing='ij')
    
    u, v, w = step_data["current_u"][x,y,z], step_data["current_v"][x,y,z], step_data["current_w"][x,y,z]
    
    # 坐标 +0.5，让箭头从网格的中心点射出
    x_c, y_c, z_c = x + 0.5, y + 0.5, z + 0.5
    
    # 绘制矢量箭头，length 控制比例缩放
    ax.quiver(x_c, y_c, z_c, u, v, w, color='c', length=0.4, arrow_length_ratio=0.3, alpha=0.9)
    
    # --- C. 绘制动态障碍物 (Scatter 红球) ---
    dyn_obs = step_data["dynamic_obstacles"]
    if dyn_obs:
        dx = [p[0] + 0.5 for p in dyn_obs]
        dy = [p[1] + 0.5 for p in dyn_obs]
        dz =[p[2] + 0.5 for p in dyn_obs]
        ax.scatter(dx, dy, dz, color='red', s=120, edgecolors='black', depthshade=True, zorder=10)

    # --- D. 坐标轴与视角美化 ---
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_zlim(0, grid_size)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Depth (m)')
    
    # 设置 2.5D 俯视等距视角
    ax.view_init(elev=25, azim=-55)
    
    # 自定义图例
    legend_elements =[
        Patch(facecolor='#4682B455', edgecolor='#333333', label='Static Terrain (Hard Constraint)'),
        Line2D([0], [0], marker='o', color='w', label='Dynamic Obstacles (Moving)', markerfacecolor='red', markersize=12, markeredgecolor='k'),
        Line2D([0], [0], color='cyan', lw=2, label='Ocean Current Vector (Soft Constraint)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.0, 1.05), fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close(fig)

# ===================== 主程序 =====================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "dynamic_ocean_data")
    os.makedirs(output_dir, exist_ok=True)
    
    print("⏳ 正在计算包含流场的三维海洋数据...")
    dynamic_map_data = generate_ocean_map()
    
    # 1. 生成 2.5D 静态分析图 (默认渲染 T=0 时刻的第一帧)
    img_save_path = os.path.join(output_dir, "ocean_snapshot_2_5d.png")
    generate_2_5d_snapshot(dynamic_map_data, t_index=0, output_path=img_save_path)
    print(f"✅ 2.5D 综合观测图已生成: {img_save_path}")
    
    # 2. 生成 Three.js 动态网页 (保持原有功能)
    print("✅ 动态数据准备完毕，(Three.js网页生成逻辑已省略/与之前一致，你随时可用前面提供的代码结合使用)...")
    print("-" * 50)
    print("👉 请前往文件夹查看洋流、礁石与动态障碍的三重叠加图！")