import os
import json
import numpy as np
from numba import jit

# ===================== 配置参数 =====================
GRID_SIZE = 20  # 20×20×20网格
TIME_STEPS = 10  # 10个时间步
SIGNAL_THRESHOLD = 0.2  # 通信阈值
SAFE_RADIUS = 2  # 避障安全半径（膨胀半径）


# ===================== 1. 静态基础层生成（地形） =====================
def generate_static_terrain(grid_size):
    # 【修复】使用 1% 的概率生成原始礁石，防止膨胀后 100% 堵死地图
    terrain = np.random.choice(
        [0, 1], (grid_size, grid_size, grid_size), p=[0.99, 0.01]
    )
    expanded_terrain = terrain.copy()

    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                if terrain[x, y, z] == 1:
                    # 按照 SAFE_RADIUS 进行膨胀
                    for dx in range(-SAFE_RADIUS, SAFE_RADIUS + 1):
                        for dy in range(-SAFE_RADIUS, SAFE_RADIUS + 1):
                            for dz in range(-SAFE_RADIUS, SAFE_RADIUS + 1):
                                nx, ny, nz = x + dx, y + dy, z + dz
                                if (
                                    0 <= nx < grid_size
                                    and 0 <= ny < grid_size
                                    and 0 <= nz < grid_size
                                ):
                                    expanded_terrain[nx, ny, nz] = 1
    return expanded_terrain


# ===================== 2. 动态环境层生成（洋流+动态障碍） =====================
@jit(nopython=True)
def generate_dynamic_current(grid_size, t):
    u = np.zeros((grid_size, grid_size, grid_size))
    v = np.zeros((grid_size, grid_size, grid_size))
    w = np.zeros((grid_size, grid_size, grid_size))
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                u[x, y, z] = 1.5 * np.sin(0.2 * x + 0.1 * t)
                v[x, y, z] = 1.5 * np.cos(0.2 * y + 0.1 * t)
                w[x, y, z] = 0.3 * np.sin(0.1 * z + 0.05 * t)
    return u, v, w


def generate_dynamic_obstacles(grid_size, t, num_obstacles=12):
    obstacles = []
    # 【修复】使用局部随机数状态，确保每次运行的动态轨迹是可复现的
    rng = np.random.RandomState(t * 10)
    for _ in range(num_obstacles):
        # 让障碍物有一个伪随机的运动轨迹
        x = (
            rng.randint(SAFE_RADIUS, grid_size - SAFE_RADIUS) + int(t * 0.8)
        ) % grid_size
        y = (
            rng.randint(SAFE_RADIUS, grid_size - SAFE_RADIUS) + int(t * 0.5)
        ) % grid_size
        z = rng.randint(SAFE_RADIUS, grid_size - SAFE_RADIUS)
        obstacles.append([int(x), int(y), int(z)])
    return obstacles


# ===================== 3. 约束掩码层生成（信号强度） =====================
@jit(nopython=True)
def generate_signal_field(grid_size, comm_point=(5, 5, 2), threshold=SIGNAL_THRESHOLD):
    signal = np.zeros((grid_size, grid_size, grid_size))
    signal_mask = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    cx, cy, cz = comm_point

    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
                val = np.exp(-0.08 * dist) * (1 - 0.02 * z)
                signal[x, y, z] = max(val, 0.0)

                if signal[x, y, z] < threshold:
                    signal_mask[x, y, z] = 100.0
                else:
                    signal_mask[x, y, z] = 0.0

    return signal, signal_mask


# ===================== 4. 整合为时空序列化地图 =====================
def generate_ocean_map():
    static_terrain = generate_static_terrain(GRID_SIZE)

    # 计算堵塞率
    blockage_rate = np.mean(static_terrain) * 100
    print(
        f"📊 静态地图堵塞率: {blockage_rate:.2f}% (建议保持在5%~20%之间，否则A*易无解)"
    )

    dynamic_map = []

    for t in range(TIME_STEPS):
        cur_u, cur_v, cur_w = generate_dynamic_current(GRID_SIZE, t)
        obstacles = generate_dynamic_obstacles(GRID_SIZE, t)
        signal, signal_mask = generate_signal_field(GRID_SIZE)

        step_map = {
            "time_step": t,
            "static_terrain": static_terrain,
            "current_u": cur_u,
            "current_v": cur_v,
            "current_w": cur_w,
            "dynamic_obstacles": obstacles,
            "signal_strength": signal,
            "signal_mask": signal_mask,
            "safe_radius": SAFE_RADIUS,
        }
        dynamic_map.append(step_map)

    return dynamic_map


# ===================== 5. 将动态数据打包为 Three.js 网页 =====================
def export_dynamic_map_to_threejs(dynamic_map, output_path):
    # 提取静态地形坐标
    static_terrain = dynamic_map[0]["static_terrain"]
    static_obs_coords = np.argwhere(static_terrain == 1).tolist()

    # 提取每一帧的动态障碍物坐标
    dyn_obs_per_step = [step["dynamic_obstacles"] for step in dynamic_map]

    # 构造注入前端的 JSON 数据
    scene_data = {
        "grid_size": GRID_SIZE,
        "time_steps": TIME_STEPS,
        "static_obstacles": static_obs_coords,
        "dynamic_obstacles_per_step": dyn_obs_per_step,
    }

    json_data = json.dumps(scene_data)

    # Three.js 前端模板 (包含时间轴动画逻辑)
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Dynamic Ocean 3D Map</title>
        <style>
            body { margin: 0; overflow: hidden; background-color: #121212; font-family: Arial, sans-serif;}
            #ui {
                position: absolute; top: 15px; left: 15px;
                color: #00FFCC; background: rgba(0,0,0,0.8);
                padding: 15px 25px; border-radius: 8px; user-select: none;
                pointer-events: none; border: 1px solid #00FFCC;
            }
            #instructions {
                position: absolute; bottom: 15px; left: 15px;
                color: #aaa; background: rgba(0,0,0,0.5);
                padding: 10px; border-radius: 5px; font-size: 12px; pointer-events: none;
            }
            .time-display { font-size: 24px; font-weight: bold; color: #FF4500; margin-top: 10px;}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <div id="ui">
            <h2 style="margin: 0; font-size: 18px;">🌊 AUV 动态海洋环境</h2>
            <div style="font-size: 14px; margin-top: 8px; color: #fff;">
                Map Size: __GRID_SIZE__ x __GRID_SIZE__ x __GRID_SIZE__ <br>
                Total Time Steps: __TIME_STEPS__
            </div>
            <div class="time-display">⏱️ 当前时间步 (T): <span id="t_val">0</span></div>
        </div>
        <div id="instructions">🖱️ 左键：旋转 | 🖱️ 右键：平移 | ⚙️ 滚轮：缩放 | 🔴 橘红色圆球代表动态移动的障碍物</div>
        
        <script>
            const SCENE_DATA = __DATA_PLACEHOLDER__;
            
            // === 初始化 Scene ===
            const scene = new THREE.Scene();
            THREE.Object3D.DefaultUp.set(0, 0, 1);
            
            const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(-15, -15, 25);
            
            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.body.appendChild(renderer.domElement);
            
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            const centerOffset = SCENE_DATA.grid_size / 2;
            controls.target.set(centerOffset, centerOffset, centerOffset/2);
            controls.update();

            // 光源
            scene.add(new THREE.AmbientLight(0xffffff, 0.5));
            const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
            dirLight.position.set(20, 20, 40);
            scene.add(dirLight);

            // 绘制网格线和水面
            const gridHelper = new THREE.GridHelper(SCENE_DATA.grid_size, SCENE_DATA.grid_size, 0x444444, 0x222222);
            gridHelper.rotation.x = Math.PI / 2;
            gridHelper.position.set(centerOffset, centerOffset, 0);
            scene.add(gridHelper);
            
            const waterGeo = new THREE.PlaneGeometry(SCENE_DATA.grid_size, SCENE_DATA.grid_size);
            const waterMat = new THREE.MeshBasicMaterial({color: 0x006994, transparent: true, opacity: 0.15, side: THREE.DoubleSide});
            const water = new THREE.Mesh(waterGeo, waterMat);
            water.position.set(centerOffset, centerOffset, SCENE_DATA.grid_size);
            scene.add(water);

            // === 绘制静态地形 (深色半透明方块) ===
            const obsGeo = new THREE.BoxGeometry(0.9, 0.9, 0.9);
            const obsMat = new THREE.MeshPhongMaterial({ color: 0x4682B4, transparent: true, opacity: 0.3 });
            const instancedMesh = new THREE.InstancedMesh(obsGeo, obsMat, SCENE_DATA.static_obstacles.length);
            
            const dummy = new THREE.Object3D();
            SCENE_DATA.static_obstacles.forEach((obs, i) => {
                dummy.position.set(obs[0] + 0.5, obs[1] + 0.5, obs[2] + 0.5);
                dummy.updateMatrix();
                instancedMesh.setMatrixAt(i, dummy.matrix);
            });
            scene.add(instancedMesh);

            // === 绘制动态障碍物 (高亮红色球体) ===
            const numDynObs = SCENE_DATA.dynamic_obstacles_per_step[0].length;
            const dynObsMeshes =[];
            const dynGeo = new THREE.SphereGeometry(0.6, 16, 16);
            const dynMat = new THREE.MeshPhongMaterial({ color: 0xFF4500 }); // 橘红色
            
            for(let i=0; i < numDynObs; i++) {
                const mesh = new THREE.Mesh(dynGeo, dynMat);
                scene.add(mesh);
                dynObsMeshes.push(mesh);
            }

            // === 开启时间轴动画循环 ===
            let currentStep = 0;
            
            // 每 800 毫秒切换一次时间步 (模拟时间流逝)
            setInterval(() => {
                const currentObs = SCENE_DATA.dynamic_obstacles_per_step[currentStep];
                
                // 更新每个动态障碍物的位置
                for(let i=0; i < numDynObs; i++) {
                    if(currentObs[i]) {
                        dynObsMeshes[i].position.set(
                            currentObs[i][0] + 0.5, 
                            currentObs[i][1] + 0.5, 
                            currentObs[i][2] + 0.5
                        );
                    }
                }
                
                // 更新 UI
                document.getElementById('t_val').innerText = currentStep;
                
                // 推进时间步
                currentStep = (currentStep + 1) % SCENE_DATA.time_steps;
                
            }, 800); // 800ms的播放速度，可自行调节

            // 渲染循环
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

    # 替换变量并写入文件
    html_content = html_template.replace("__DATA_PLACEHOLDER__", json_data)
    html_content = html_content.replace("__GRID_SIZE__", str(GRID_SIZE))
    html_content = html_content.replace("__TIME_STEPS__", str(TIME_STEPS))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


# ===================== 主程序 =====================
if __name__ == "__main__":
    # 1. 自动创建数据保存文件夹
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "dynamic_ocean_data")
    os.makedirs(output_dir, exist_ok=True)

    # 2. 生成多时间步动态地图
    print("-" * 50)
    print("⏳ 正在生成带约束的 3D 动态海洋地图...")
    dynamic_map_data = generate_ocean_map()

    # 3. 将数据保存为 Numpy 格式，供后续训练/算法读取
    npy_save_path = os.path.join(output_dir, "ocean_dynamic_map.npy")
    # 【修复】增加 dtype=object 消除 Numpy 保存字典列表时的警告
    np.save(npy_save_path, np.array(dynamic_map_data, dtype=object))
    print(f"✅ Numpy 数据已保存: {npy_save_path}")

    # 4. 生成 3D 可视化 HTML 文件
    html_save_path = os.path.join(output_dir, "dynamic_ocean_visualization.html")
    export_dynamic_map_to_threejs(dynamic_map_data, html_save_path)
    print(f"✅ 3D 可视化网页已生成: {html_save_path}")
    print("-" * 50)
    print("👉 去文件夹中双击打开 html 文件，即可观看带时间轴动画的动态地图！")
