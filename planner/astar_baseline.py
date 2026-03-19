import os
import time
import json
import numpy as np
from numba import jit

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

# ==================== 生成 HTML/Three.js 交互文件的核心代码 ====================

def export_to_threejs(grid, path, start, goal, visited, title, output_path):
    # 提取障碍物坐标
    obstacles = np.argwhere(grid == 1).tolist()
    
    # 构建数据字典
    scene_data = {
        "title": title,
        "shape": list(grid.shape),
        "start": start,
        "goal": goal,
        "path": path if path else[],
        "obstacles": obstacles,
        "stats": {
            "path_length": len(path) if path else 0,
            "visited_nodes": visited
        }
    }
    
    # 序列化为 JSON 字符串
    json_data = json.dumps(scene_data)
    
    # 纯正的前端 HTML 模板 (包含 Three.js 和 OrbitControls)
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>AUV 3D Path Planning</title>
        <style>
            body { margin: 0; overflow: hidden; background-color: #1e1e1e; font-family: Arial, sans-serif;}
            #ui {
                position: absolute; top: 15px; left: 15px;
                color: white; background: rgba(0,0,0,0.7);
                padding: 15px; border-radius: 8px; user-select: none;
                pointer-events: none; border: 1px solid #444;
            }
            #instructions {
                position: absolute; bottom: 15px; left: 15px;
                color: #aaa; background: rgba(0,0,0,0.5);
                padding: 10px; border-radius: 5px; font-size: 12px; pointer-events: none;
            }
        </style>
        <!-- 引入 Three.js -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <!-- 引入 轨道控制器 -->
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <div id="ui">
            <h2 id="title" style="margin: 0 0 10px 0; font-size: 18px; color: #4dc3ff;"></h2>
            <div id="stats" style="font-size: 14px; line-height: 1.5;"></div>
        </div>
        <div id="instructions">🖱️ 鼠标左键：旋转视角 | 🖱️ 鼠标右键：平移视角 | ⚙️ 滚轮：缩放</div>
        
        <script>
            // === 接收 Python 传来的数据 ===
            const SCENE_DATA = __DATA_PLACEHOLDER__;
            
            // 更新 UI
            document.getElementById('title').innerText = SCENE_DATA.title;
            document.getElementById('stats').innerHTML = `
                📌 Map Size: ${SCENE_DATA.shape[0]} x ${SCENE_DATA.shape[1]} x ${SCENE_DATA.shape[2]} <br>
                🔍 Visited Nodes: ${SCENE_DATA.stats.visited_nodes} <br>
                📏 Path Length: ${SCENE_DATA.stats.path_length} <br>
                🟢 Start: [${SCENE_DATA.start}] <br>
                ⭐ Goal: [${SCENE_DATA.goal}]
            `;

            // === 1. 初始化 Three.js 场景 ===
            const scene = new THREE.Scene();
            // 设置 Z 轴向上 (贴合数学系坐标)
            THREE.Object3D.DefaultUp.set(0, 0, 1);
            
            const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(-20, -20, 35); // 初始化相机位置
            
            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.body.appendChild(renderer.domElement);
            
            // 控制器
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.target.set(SCENE_DATA.shape[0]/2, SCENE_DATA.shape[1]/2, SCENE_DATA.shape[2]/3);
            controls.update();

            // 光源
            scene.add(new THREE.AmbientLight(0xffffff, 0.6));
            const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
            dirLight.position.set(20, 20, 40);
            scene.add(dirLight);

            // === 2. 绘制网格线和水面 ===
            const gridHelper = new THREE.GridHelper(SCENE_DATA.shape[0], SCENE_DATA.shape[0], 0x444444, 0x222222);
            gridHelper.rotation.x = Math.PI / 2; // 旋转使其平行于 XY 平面
            gridHelper.position.set(SCENE_DATA.shape[0]/2, SCENE_DATA.shape[1]/2, 0);
            scene.add(gridHelper);
            
            // 水面/海平面 (Z = max Z)
            const waterGeo = new THREE.PlaneGeometry(SCENE_DATA.shape[0], SCENE_DATA.shape[1]);
            const waterMat = new THREE.MeshBasicMaterial({color: 0x006994, transparent: true, opacity: 0.15, side: THREE.DoubleSide});
            const water = new THREE.Mesh(waterGeo, waterMat);
            water.position.set(SCENE_DATA.shape[0]/2, SCENE_DATA.shape[1]/2, SCENE_DATA.shape[2]);
            scene.add(water);

            // === 3. 使用 InstancedMesh 高效渲染海量障碍物体素 ===
            const obsGeo = new THREE.BoxGeometry(0.95, 0.95, 0.95); // 稍微缩小一点产生网格缝隙感
            const obsMat = new THREE.MeshPhongMaterial({ color: 0x4682B4, transparent: true, opacity: 0.4 });
            const instancedMesh = new THREE.InstancedMesh(obsGeo, obsMat, SCENE_DATA.obstacles.length);
            
            const dummy = new THREE.Object3D();
            SCENE_DATA.obstacles.forEach((obs, i) => {
                dummy.position.set(obs[0] + 0.5, obs[1] + 0.5, obs[2] + 0.5);
                dummy.updateMatrix();
                instancedMesh.setMatrixAt(i, dummy.matrix);
            });
            scene.add(instancedMesh);

            // === 4. 绘制路径、起点和终点 ===
            function createSphere(pos, color, size=0.8) {
                const geo = new THREE.SphereGeometry(size, 16, 16);
                const mat = new THREE.MeshPhongMaterial({ color: color });
                const mesh = new THREE.Mesh(geo, mat);
                mesh.position.set(pos[0] + 0.5, pos[1] + 0.5, pos[2] + 0.5);
                scene.add(mesh);
            }
            
            // 绘制起点和终点
            createSphere(SCENE_DATA.start, 0x00FFFF, 1.2); // 青色起点
            createSphere(SCENE_DATA.goal, 0xFFD700, 1.2);  // 金色终点

            // 绘制路径线段和路径节点
            if (SCENE_DATA.path.length > 0) {
                const points = SCENE_DATA.path.map(p => new THREE.Vector3(p[0] + 0.5, p[1] + 0.5, p[2] + 0.5));
                
                // 红色的连线
                const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
                const lineMat = new THREE.LineBasicMaterial({ color: 0xFF0000, linewidth: 2 });
                const line = new THREE.Line(lineGeo, lineMat);
                scene.add(line);
                
                // 在每个路径点画一个小圆球以增强 3D 可视度
                SCENE_DATA.path.forEach(p => {
                    createSphere(p, 0xFF0000, 0.2);
                });
            }

            // === 5. 渲染循环 ===
            window.addEventListener('resize', () => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            });

            function animate() {
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }
            animate();
        </script>
    </body>
    </html>
    """
    
    # 替换占位符并保存
    html_content = html_template.replace('__DATA_PLACEHOLDER__', json_data)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def run_and_export():
    shape = (30, 30, 20)
    start = (2, 2, 18)   
    goal = (27, 27, 2)   
    
    def clear_start_goal(g):
        g[0:4, 0:4, 16:20] = 0
        g[25:30, 25:30, 0:4] = 0
        return g

    # 准备地图
    grid_A = clear_start_goal(create_reef_and_minefield(shape))
    grid_B = clear_start_goal(create_subsea_infrastructure(shape))

    scenarios =[
        ("Scenario A: Underwater Reefs & Minefield", grid_A, "scenario_A_interactive.html"),
        ("Scenario B: Subsea Infrastructure & Seabed", grid_B, "scenario_B_interactive.html")
    ]

    # 创建自动保存文件夹
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_dir = os.path.join(script_dir, f"{script_name}_results")
    os.makedirs(output_dir, exist_ok=True)
    
    print("-" * 50)
    print("🚀 开始 A* 路径规划并生成 Three.js 3D 交互网页 ...")

    for title, grid, filename in scenarios:
        print(f"\n🌊 正在规划 {title} ...")
        t0 = time.time()
        path, visited = astar_3d(grid, start, goal)
        t1 = time.time()
        
        if path:
            print(f"✅ 找到路径！耗时: {(t1-t0)*1000:.2f} ms | 准备打包成网页...")
        
        # 导出为 HTML
        save_path = os.path.join(output_dir, filename)
        export_to_threejs(grid, path, start, goal, visited, title, save_path)
        print(f"📄 已生成: {save_path}")

    print("\n🎉 全部完成！")
    print("👉 请前往生成的文件夹中，双击 .html 文件即可在浏览器中体验丝滑的 3D 视角！")
    print("-" * 50)

if __name__ == "__main__":
    run_and_export()