import os
import time
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义3D移动方向（6个方向：上下左右前后）
DIRECTIONS = np.array([(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)])

@jit(nopython=True)
def heuristic_3d(a, b):
    """3D曼哈顿距离作为A*启发函数h(n)"""
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])

@jit(nopython=True)
def astar_3d(grid, start, goal):
    """3D A*核心算法（Numba加速）"""
    empty_path = [(0, 0, 0)]
    empty_path.clear()

    x_size, y_size, z_size = grid.shape
    if (start[0]<0 or start[0]>=x_size or start[1]<0 or start[1]>=y_size or start[2]<0 or start[2]>=z_size or grid[start]==1):
        return empty_path, 0
    if (goal[0]<0 or goal[0]>=x_size or goal[1]<0 or goal[1]>=y_size or goal[2]<0 or goal[2]>=z_size or grid[goal]==1):
        return empty_path, 0
    
    open_list = [(0.0, start[0], start[1], start[2])]
    closed_set = np.zeros(grid.shape, dtype=np.bool_)
    g_score = np.full(grid.shape, np.inf)
    g_score[start] = 0
    f_score = np.full(grid.shape, np.inf)
    f_score[start] = heuristic_3d(start, goal)
    came_from = np.empty(grid.shape + (3,), dtype=np.int32)
    came_from[:] = -1
    visited_nodes = 0

    while open_list:
        open_list.sort()
        current_f, x, y, z = open_list.pop(0)
        current_pos = (x, y, z)
        
        if closed_set[current_pos]:
            continue
        closed_set[current_pos] = True
        visited_nodes += 1

        if current_pos == goal:
            path = [(0, 0, 0)]
            path.clear()
            while current_pos != (-1, -1, -1):
                path.append(current_pos)
                px, py, pz = came_from[current_pos]
                current_pos = (px, py, pz)
            path.reverse()
            return path, visited_nodes
        
        for dx, dy, dz in DIRECTIONS:
            nx, ny, nz = x+dx, y+dy, z+dz
            if (nx<0 or nx>=x_size or ny<0 or ny>=y_size or nz<0 or nz>=z_size or grid[nx,ny,nz]==1 or closed_set[nx,ny,nz]):
                continue
            tentative_g = g_score[x,y,z] + 1.0
            if tentative_g < g_score[nx,ny,nz]:
                came_from[nx,ny,nz] = (x,y,z)
                g_score[nx,ny,nz] = tentative_g
                f_score[nx,ny,nz] = tentative_g + heuristic_3d((nx,ny,nz), goal)
                open_list.append((f_score[nx,ny,nz], nx, ny, nz))
                
    return empty_path, visited_nodes

# ==================== 海底复杂地图生成函数 ====================

def create_reef_and_minefield(shape=(30, 30, 20)):
    """场景A：水下暗礁与水雷区（密集随机散布+局部聚集）"""
    grid = np.zeros(shape, dtype=np.int32)
    x_s, y_s, z_s = shape
    
    # 生成几十个随机簇（模拟成片的暗礁或雷区）
    for _ in range(60):
        cx, cy, cz = np.random.randint(0, x_s), np.random.randint(0, y_s), np.random.randint(0, z_s)
        # 障碍簇的大小
        sx, sy, sz = np.random.randint(1, 4), np.random.randint(1, 4), np.random.randint(1, 3)
        grid[max(0, cx-sx):min(x_s, cx+sx), max(0, cy-sy):min(y_s, cy+sy), max(0, cz-sz):min(z_s, cz+sz)] = 1
        
    return grid

def create_subsea_infrastructure(shape=(30, 30, 20)):
    """场景B：水下油气平台与起伏海床（复杂地形+垂直桩基）"""
    grid = np.zeros(shape, dtype=np.int32)
    x_s, y_s, z_s = shape
    
    # 1. 模拟起伏的海床 (利用正弦波+余弦波生成自然地貌)
    for x in range(x_s):
        for y in range(y_s):
            h = int((np.sin(x / 4.0) + np.cos(y / 3.0)) * 2 + 4) 
            h = max(0, min(h, z_s - 1))
            grid[x, y, 0:h] = 1 
            
    # 2. 模拟海底油气平台的垂直桩基 (从海底直通水面)
    platform_centers =[(10, 10), (10, 20), (20, 10), (20, 20)]
    for px, py in platform_centers:
        grid[px:px+2, py:py+2, :] = 1 
        
    # 3. 模拟悬浮的管线/横梁 (连接各桩基的横向结构)
    grid[10:21, 10, 12] = 1
    grid[10:21, 20, 12] = 1
    grid[10, 10:21, 8] = 1
    grid[20, 10:21, 8] = 1
        
    return grid

# ==================== 可视化对比测试 ====================

def visualize_subsea_comparison():
    shape = (30, 30, 20)
    start = (2, 2, 18)   # 左上角，靠近水面
    goal = (27, 27, 2)   # 右下角，靠近海底
    
    # 强制清空起点和终点的周围，确保可通行
    def clear_start_goal(g):
        g[0:4, 0:4, 16:20] = 0
        g[25:30, 25:30, 0:4] = 0
        return g

    grid_A = clear_start_goal(create_reef_and_minefield(shape))
    grid_B = clear_start_goal(create_subsea_infrastructure(shape))

    maps =[
        ("Scenario A: Underwater Reefs & Minefield", grid_A),
        ("Scenario B: Subsea Infrastructure & Seabed", grid_B)
    ]

    fig = plt.figure(figsize=(18, 8))
    
    for i, (title, grid) in enumerate(maps):
        print(f"\n🌊 正在规划 {title} ...")
        t0 = time.time()
        path, visited = astar_3d(grid, start, goal)
        t1 = time.time()
        
        if path:
            print(f"✅ 找到路径！耗时: {(t1-t0)*1000:.2f} ms | 路径长度: {len(path)} | 搜索节点数: {visited}")
        else:
            print("❌ 无解！")

        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        ax.set_title(f"{title}\nNodes visited: {visited} | Path length: {len(path)}", fontsize=12)
        
        obs_x, obs_y, obs_z = np.where(grid == 1)
        ax.scatter(obs_x, obs_y, obs_z, c='gray', s=10, alpha=0.15, label='Obstacles/Terrain')
        
        if path:
            path_x, path_y, path_z = zip(*path)
            ax.plot(path_x, path_y, path_z, c='red', linewidth=3, label='AUV Trajectory')
            ax.scatter(path_x[::5], path_y[::5], path_z[::5], c='red', s=15)
            
        ax.scatter(start[0], start[1], start[2], c='cyan', s=150, marker='o', edgecolors='black', label='AUV Start')
        ax.scatter(goal[0], goal[1], goal[2], c='gold', s=200, marker='*', edgecolors='black', label='Target Station')
        
        ax.set_xlim(0, shape[0])
        ax.set_ylim(0, shape[1])
        ax.set_zlim(0, shape[2])
        ax.set_xlabel('X (m)'), ax.set_ylabel('Y (m)'), ax.set_zlabel('Depth (m)')
        ax.view_init(elev=30, azim=-45)
        ax.legend(loc='upper left')

    # ================= 动态创建文件夹并保存图片 =================
    plt.tight_layout()
    
    # 1. 获取当前正在执行的 Python 文件所在的绝对目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. 获取当前执行的 Python 脚本的名称（不带 .py 后缀），例如 'astar_baseline'
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    
    # 3. 拼接出我们要创建的文件夹的路径，比如 'astar_baseline_results'
    output_dir = os.path.join(script_dir, f"{script_name}_results")
    
    # 4. 如果文件夹不存在，则自动创建它
    os.makedirs(output_dir, exist_ok=True)
    
    # 5. 指定图片最后要保存的位置
    img_filename = 'subsea_astar_comparison.png'
    save_path = os.path.join(output_dir, img_filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print("-" * 50)
    print(f"📂 文件夹已自动创建: {output_dir}")
    print(f"📸 3D可视化对比已生成: {img_filename}")
    print("-" * 50)

if __name__ == "__main__":
    visualize_subsea_comparison()