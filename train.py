from maze_env import MazeEnvironment
from q_learning_agent import QLearningAgent

if __name__ == "__main__":
    # 选择迷宫生成方式
    print("Select the type of maze:")
    print("1. Randomly generated maze (with walls placed randomly)")
    print("2. Fixed-layout maze")
    choice = input("Please make a selection (1 or 2, default is 1): ").strip()
    
    if choice == '2':
        print("Use Fixed-layout maze")
        env = MazeEnvironment(maze_size=(10, 10), use_random=False)
        
    else:
        print("Use Randomly generated maze (with walls placed randomly)")
        try:
            wall_prob = float(input("Input wall quantity density (0.1 - 0.5, default 0.3): ") or "0.3")
            wall_prob = max(0.1, min(0.5, wall_prob))  # 限制范围
        except:
            wall_prob = 0.3
            
        env = MazeEnvironment(maze_size=(10, 10), use_random=True, wall_probability=wall_prob)
    
    print("Maze env created！")
    
    # 显示初始迷宫
    print("Initial Maze:")
    env.render()
    
    # 创建Q-learning智能体
    agent = QLearningAgent(env, 
                          learning_rate=0.1,
                          discount_factor=0.95,
                          epsilon=1.0,
                          epsilon_decay=0.995,
                          epsilon_min=0.01)
    
    # 训练智能体（启用视频录制）
    print("Whether to record the training process as a GIF?(Y/N, default is N)")
    record_training = input().upper() == 'Y'
    
    agent.train(episodes=1000, max_steps_per_episode=200, 
                record_video=record_training, video_interval=20)
    
    # 绘制训练进度
    agent.plot_training_progress()
    
    # 测试智能体
    agent.test(num_tests=10)
    
    # 显示训练后的策略
    env.reset()
    print("\nThe trained strategy (the red arrow indicates the optimal action):")
    env.render(q_table=agent.q_table, show_values=True)
    
    # 创建GIF动画
    print("\nWhether to create GIF？")
    create_gifs = input("Input 'Y' to create GIF: ").upper() == 'Y'
    
    if create_gifs:
        # 创建寻路演示GIF
        steps_used = agent.create_pathfinding_gif("pathfinding_demo.gif", fps=3)
        
        # 如果有训练数据，创建训练过程GIF
        if record_training and agent.training_positions:
            agent.create_training_gif("training_process.gif", fps=15)
        
        print("The generated file：")
        print("- assets/pathfinding_demo.gif")
        if record_training and agent.training_positions:
            print("- assets/training_process.gif")
    
    # 展示一次完整的路径
    print("\nShow a complete path-finding process:")
    state = env.reset()
    path = [state]
    
    for step in range(50):  # 最多50步
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        path.append(next_state)
        state = next_state
        
        if done:
            print(f"Path Founded！ {step + 1} Steps Total")
            print(f"Path: {' -> '.join([str(p) for p in path])}")
            break
    
    env.render(q_table=agent.q_table, show_values=True)