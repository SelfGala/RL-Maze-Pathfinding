# 迷宫强化学习项目

这个项目使用Q-learning算法训练智能体在迷宫中寻找最优路径，并生成GIF动画展示训练过程和最终寻路结果。

## 文件结构

```
.
├── maze_environment.py    # 迷宫环境类
├── qlearning_agent.py     # Q-learning智能体类
├── main.py               # 主程序
├── requirements.txt      # 依赖包列表
└── README.md            # 说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行程序

```bash
python main.py
```

## 功能特点

### 迷宫环境 (MazeEnvironment)
- 生成10x10的迷宫，包含墙壁、起点和终点
- 支持智能体的移动和碰撞检测
- 提供奖励机制：到达目标+100，撞墙-1，每步-0.1
- 可视化迷宫和智能体位置

### Q-learning智能体 (QLearningAgent)  
- 使用epsilon-greedy策略进行探索与利用的平衡
- 动态调整学习参数
- 记录训练统计数据
- 生成训练过程和寻路演示的GIF动画

### 主要功能
1. **训练智能体**：1000轮训练，自动调整探索率
2. **可视化训练进度**：显示奖励和步数的变化曲线
3. **测试性能**：10次测试评估智能体表现
4. **策略可视化**：用红色箭头显示学习到的最佳动作
5. **GIF动画生成**：
   - `training_process.gif`: 训练过程动画
   - `pathfinding_demo.gif`: 最终寻路演示

## 使用说明

1. 运行程序后，首先会显示初始迷宫
2. 选择是否录制训练过程（输入y或n）
3. 程序会自动进行1000轮训练
4. 训练完成后显示训练进度图表
5. 进行10次测试并显示结果
6. 选择是否生成GIF动画
7. 最后展示一次完整的寻路过程

## 参数说明

- `learning_rate`: 学习率 (0.1)
- `discount_factor`: 折扣因子 (0.95)  
- `epsilon`: 初始探索率 (1.0)
- `epsilon_decay`: 探索率衰减 (0.995)
- `epsilon_min`: 最小探索率 (0.01)

## 输出文件

- `pathfinding_demo.gif`: 智能体寻路演示动画
- `training_process.gif`: 训练过程动画（如果选择录制）

## 注意事项

- 确保已安装Pillow库用于GIF生成
- GIF文件大小取决于训练数据量，可调整`video_interval`参数控制录制频率
- 迷宫布局在`MazeEnvironment.generate_maze()`方法中定义，可自定义修改