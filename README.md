# 微调流程
环境：ubuntu22.04, ros2 humble  
前置：安装 [π₀](https://github.com/Physical-Intelligence/openpi)官方库中的环境
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## 收集数据：pipeline中的task
### 准备
从[realsense官方库](https://github.com/IntelRealSense/realsense-ros)安装驱动后使用命令启动相机节点
```bash
ros2 launch realsense2_camera rs_launch.py   rgb_camera.color_profile:=640,480,30 camera_name:=outside
```
由于realman机械臂固件未升级，最新的api与官方ros库均无法使用，只能用旧版本api  
```bash
uv pip install Robotic_Arm==1.0.1
```

安装松灵底盘机器人ranger官方[ros库](https://github.com/agilexrobotics/ranger_ros2)并配置，完成后使用命令启动
```bash
# 以下为非首次启动的指令，首次启动略有不同
sudo ip link set can0 down # can0可能繁忙 先关闭can0 
sudo ip link set can0 up type can bitrate 500000
ros2 launch ranger_bringup ranger_mini_v3.launch.xml
```

### 数据收集方案
预想从起点出发到达目标双臂执行之后返回起点，整体全自动；但如果小车不沿着直线运动，误差会较大一点导致机械臂无法固定点位抓取，因此目前只沿着直线运动，即从起点直线运动到终点执行抓取，再返回起点并且人工校准起点减少误差,详见[task](pipeline/task.py)

## 数据预处理
[rosbag转hdf5](pipeline/db3_to_hdf5.py) 

[hdf5角度转弧度并增加actions](pipeline/process_hdf5.py)  

[hdf5转换为lerobot格式](pipeline/hdf5_to_lerobot.py)  

## 配置训练信息
[下载模型权重](utils/hf_download.py)，选择pi_base或者pi_fast_base

需要设置自己的policy处理数据的输入输出  
并在config配置自己的dataconfig和trainconfig  

计算norm信息
```bash
uv run scripts/compute_norm_stats.py --config-name pi0_fast_ranger
```
启动训练
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_base_ranger --exp-name=0613 --overwrite
```
## 推理
启动[推理服务器](pipeline/inference_server.py)，[客户端](pipeline/inference_client.py)连接机器人向服务器发送请求获取推理结果