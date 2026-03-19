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
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])

@jit(nopython=True)
def astar_3d(grid, start, goal):
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

# ==================== 海底复杂地图生成 ====================

def create_reef_and_minefield(shape=(30, 30, 20)):
    grid = np.zeros(shape, dtype=np.int32)
    x_s, y_s, z_s = shape
    for _ in range(60):
        cx, cy, cz = np.random.randint(0, x_s), np.random.randint(0, y_s), np.random.randint(0, z_s)
        sx, sy, sz = np.random.randint(1, 4), np.random.randint(1, 4), np.random.randint(1, 3)
        grid[max(0, cx-sx):min(x_s, cx+sx), max(0, cy-sy):min(y_s, cy+sy), max(0, cz-sz):min(z_s, cz+sz)] = 1
    return grid

def create_subsea_infrastructure(shape=(30, 30, 20)):
    grid = np.zeros(shape, dtype=np.int32)
    x_s, y_s, z_s = shape
    for x in range(x_s):
        for y in range(y_s):
            h = int((np.sin(x / 4.0) + np.cos(y / 3.0)) * 2 + 4) 
            h = max(0, min(h, z_s - 1))
            grid[x, y, 0:h] = 1 
    platform_centers =[(10, 10), (10, 20), (20, 10), (20, 20)]
    for px, py in platform_centers:
        grid[px:px+2, py:py+2, :] = 1 
    grid[10:21, 10, 12] = 1
    grid[10:21, 20, 12] = 1
    grid[10, 10:21, 8] = 1
    grid[20, 10:21, 8] = 1
    return grid

# ==================== 优化的可视化渲染 ====================

def visualize_subsea_comparison():
    shape = (30, 30, 20)
    start = (2, 2, 18)   
    goal = (27, 27, 2)   
    
    def clear_start_goal(g):
        g[0:4, 0:4, 16:20] = 0
        g[25:30, 25:30, 0:4] = 0
        return g

    grid_A = clear_start_goal(create_reef_and_minefield(shape))
    grid_B = clear_start_goal(create_subsea_infrastructure(shape))

    maps =[
        ("Scenario A: Underwater Reefs & Minefield", grid_A, '#2E8B5733'), # 半透明海绿色
        ("Scenario B: Subsea Infrastructure & Seabed", grid_B, '#4682B444')  # 半透明钢蓝色
    ]

    fig = plt.figure(figsize=(20, 9))
    
    for i, (title, grid, color_hex) in enumerate(maps):
        print(f"\n🌊 正在规划 {title} ... (渲染Voxel体素图可能需要几秒钟)")
        t0 = time.time()
        path, visited = astar_3d(grid, start, goal)
        t1 = time.time()
        
        if path:
            print(f"✅ 规划完成！耗时: {(t1-t0)*1000:.2f} ms | 渲染图像中...")
        
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        ax.set_title(f"{title}\nNodes visited: {visited} | Path length: {len(path)}", fontsize=13)
        
        # 1. 使用 Voxels (体素方块) 代替散点图
        # 把 numpy的int数组转成bool数组给voxels用
        bool_grid = grid.astype(bool)
        # 给所有的障碍物方块上色（带透明度和黑边）
        colors = np.empty(bool_grid.shape, dtype=object)
        colors[bool_grid] = color_hex
        ax.voxels(bool_grid, facecolors=colors, edgecolor='black', linewidth=0.2)
        
        if path:
            # 2. 坐标 +0.5：因为 Voxel 方块是以 [0,1] 占据空间的，+0.5让路径完美穿过方块中心
            path_x = np.array([p[0] for p in path]) + 0.5
            path_y = np.array([p[1] for p in path]) + 0.5
            path_z = np.array([p[2] for p in path]) + 0.5
            
            # 画出真实的3D路径
            ax.plot(path_x, path_y, path_z, c='red', linewidth=3.5, label='AUV Trajectory')
            ax.scatter(path_x[::5], path_y[::5], path_z[::5], c='darkred', s=20)
            
            # 3. 增加底部 2D 阴影投影（极大增强深度感）
            ax.plot(path_x, path_y, np.zeros_like(path_z), c='black', linestyle='--', linewidth=1.5, alpha=0.5, label='XY Projection (Shadow)')
            
        # 起点终点同样 +0.5 居中
        ax.scatter(start[0]+0.5, start[1]+0.5, start[2]+0.5, c='cyan', s=150, marker='o', edgecolors='black', label='Start', zorder=5)
        ax.scatter(goal[0]+0.5, goal[1]+0.5, goal[2]+0.5, c='gold', s=200, marker='*', edgecolors='black', label='Goal', zorder=5)
        
        # 统一坐标轴视野
        ax.set_xlim(0, shape[0])
        ax.set_ylim(0, shape[1])
        ax.set_zlim(0, shape[2])
        ax.set_xlabel('X (m)'), ax.set_ylabel('Y (m)'), ax.set_zlabel('Depth (m)')
        ax.view_init(elev=25, azim=-50) # 优化视角，略微降低仰角
        ax.legend(loc='upper left')

    plt.tight_layout()
    
    # 动态创建文件夹并保存图片
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_dir = os.path.join(script_dir, f"{script_name}_results")
    os.makedirs(output_dir, exist_ok=True)
    
    img_filename = 'subsea_astar_voxel_comparison.png'
    save_path = os.path.join(output_dir, img_filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print("-" * 50)
    print(f"📂 文件夹已自动创建: {output_dir}")
    print(f"📸 3D优化版(Voxel)已生成: {img_filename}")
    print("-" * 50)

if __name__ == "__main__":
    visualize_subsea_comparison()