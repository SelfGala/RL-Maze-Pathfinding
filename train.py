from maze_env import MazeEnvironment
from q_learning_agent import QLearningAgent

# 主程序
if __name__ == "__main__":
    # 选择迷宫生成方式
    print("选择迷宫类型:")
    print("1. 随机生成的迷宫（简单随机放置墙壁）")
    print("2. 固定布局的迷宫")
    choice = input("请输入选择 (1/2, 默认为1): ").strip()
    
    if choice == '2':
        print("使用固定布局的迷宫")
        env = MazeEnvironment(maze_size=(10, 10), use_random=False)
        
    else:
        print("使用简单随机生成的迷宫")
        try:
            wall_prob = float(input("输入墙壁密度 (0.1-0.5, 默认0.3): ") or "0.3")
            wall_prob = max(0.1, min(0.5, wall_prob))  # 限制范围
        except:
            wall_prob = 0.3
            
        env = MazeEnvironment(maze_size=(10, 10), use_random=True, wall_probability=wall_prob)
    
    print("迷宫环境创建完成！")
    
    # 显示初始迷宫
    print("初始迷宫:")
    env.render()
    
    # 创建Q-learning智能体
    agent = QLearningAgent(env, 
                          learning_rate=0.1,
                          discount_factor=0.95,
                          epsilon=1.0,
                          epsilon_decay=0.995,
                          epsilon_min=0.01)
    
    # 训练智能体（启用视频录制）
    print("是否录制训练过程GIF？(y/n)")
    record_training = input().lower() == 'y'
    
    agent.train(episodes=1000, max_steps_per_episode=200, 
                record_video=record_training, video_interval=20)
    
    # 绘制训练进度
    agent.plot_training_progress()
    
    # 测试智能体
    agent.test(num_tests=10)
    
    # 显示训练后的策略
    env.reset()
    print("\n训练后的策略（红色箭头表示最佳动作）:")
    env.render(q_table=agent.q_table, show_values=True)
    
    # 创建GIF动画
    print("\n是否创建GIF动画？")
    create_gifs = input("输入 'y' 创建GIF: ").lower() == 'y'
    
    if create_gifs:
        # 创建寻路演示GIF
        steps_used = agent.create_pathfinding_gif("pathfinding_demo.gif", fps=3)
        
        # 如果有训练数据，创建训练过程GIF
        if record_training and agent.training_positions:
            agent.create_training_gif("training_process.gif", fps=15)
        
        print(f"\nGIF动画创建完成！")
        print("生成的文件：")
        print("- pathfinding_demo.gif: 寻路演示动画")
        if record_training and agent.training_positions:
            print("- training_process.gif: 训练过程动画")
    
    # 展示一次完整的路径
    print("\n展示一次完整的寻路过程:")
    state = env.reset()
    path = [state]
    
    for step in range(50):  # 最多50步
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        path.append(next_state)
        state = next_state
        
        if done:
            print(f"找到路径！用了 {step + 1} 步")
            print(f"路径: {' -> '.join([str(p) for p in path])}")
            break
    
    env.render(q_table=agent.q_table, show_values=True)