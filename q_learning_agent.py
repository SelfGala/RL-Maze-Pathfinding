import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import os
from collections import defaultdict

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q Table
        self.q_table = defaultdict(lambda: np.zeros(len(env.actions)))
        
        self.episode_rewards = []
        self.episode_steps = []
        
        self.training_positions = []
        self.training_episodes = []
    
    def choose_action(self, state):
        """epsilon-greedy"""
        if random.random() < self.epsilon:
            return random.randint(0, len(self.env.actions) - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q Table"""
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        
        # Q-learning Formula
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    def train(self, episodes=1000, max_steps_per_episode=200, record_video=False, video_interval=50):
        print(f"Start the Training，{episodes} rounds total...")
        
        # 如果需要录制视频，准备数据存储
        if record_video:
            self.training_positions = []
            self.training_episodes = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            # 记录本轮的位置序列（用于视频）
            episode_positions = []
            
            for step in range(max_steps_per_episode):
                # 记录当前位置
                if record_video:
                    episode_positions.append(self.env.current_pos)
                
                # 选择动作
                action = self.choose_action(state)
                
                # 执行动作
                next_state, reward, done = self.env.step(action)
                
                # 更新Q值
                self.update_q_value(state, action, reward, next_state)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    if record_video:
                        episode_positions.append(self.env.current_pos)
                    break
            
            # 记录统计信息
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            
            # 记录训练数据用于视频
            if record_video and episode % video_interval == 0:
                self.training_positions.extend(episode_positions)
                self.training_episodes.extend([episode] * len(episode_positions))
            
            # 衰减epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # 打印进度
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_steps = np.mean(self.episode_steps[-100:])
                print(f"Episode {episode + 1}/{episodes}, "
                      f"Average rewards: {avg_reward:.2f}, "
                      f"Average steps: {avg_steps:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
    
    def test(self, num_tests=10):
        test_epsilon = self.epsilon
        self.epsilon = 0  # 测试时不探索
        
        success_count = 0
        total_steps = 0
        
        for test in range(num_tests):
            state = self.env.reset()
            steps = 0
            
            for step in range(200):  # 最多200步
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                state = next_state
                steps += 1
                
                if done:
                    success_count += 1
                    total_steps += steps
                    print(f"Test {test + 1}: Win！ {steps} steps total")
                    break
            else:
                print(f"Test {test + 1}: Fail")
        
        self.epsilon = test_epsilon  # 恢复原epsilon值
        
        if success_count > 0:
            avg_steps = total_steps / success_count
            print(f"\nTest: {success_count}/{num_tests} Win！")
            print(f"Average steps: {avg_steps:.1f}")
        else:
            print(f"\nTest: {success_count}/{num_tests} Win！")
    
    def plot_training_progress(self):
        """绘制训练进度图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制奖励曲线
        ax1.plot(self.episode_rewards)
        ax1.set_title('Train Rewards per Episode')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True)
        
        # 绘制步数曲线
        ax2.plot(self.episode_steps)
        ax2.set_title('Train Steps per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def create_training_gif(self, filename="training_process.gif", fps=20):
        """创建Training的GIF"""
        
        assets_dir = "assets"
        if not os.path.exists(assets_dir):
            os.makedirs(assets_dir)
            
        full_path = os.path.join(assets_dir, filename)
        
        print(f"Creating tarining GIF: {full_path}")
        if not self.training_positions:
            print("No training data available to create GIF. Set record_video=True")
            return
        
        print(f"The training process GIF is being created: {full_path}")
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 迷宫显示
        maze_display = self.env.get_maze_for_animation()
        cmap = plt.cm.colors.ListedColormap(['white', 'black', 'green', 'red'])
        im = ax.imshow(maze_display, cmap=cmap, interpolation='nearest')
        
        # 初始化智能体位置
        agent_dot, = ax.plot([], [], 'bo', markersize=12)
        
        title = ax.set_title('')
        ax.set_xticks(range(self.env.cols))
        ax.set_yticks(range(self.env.rows))
        ax.grid(True, alpha=0.3)
        
        def animate(frame):
            if frame < len(self.training_positions):
                pos = self.training_positions[frame]
                episode = self.training_episodes[frame]
                
                # 更新智能体位置
                agent_dot.set_data([pos[1]], [pos[0]])
                
                title.set_text(f'Process - Episode: {episode}, Step: {frame}')
                
                return agent_dot, title
            return agent_dot, title
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=len(self.training_positions),
                                     interval=1000//fps, blit=False, repeat=True)
        
        # 保存为GIF
        try:
            anim.save(full_path, writer='pillow', fps=fps)
            print(f"Training process GIF saved as: {full_path}")
        except Exception as e:
            print(f"GIF saved failed: {e}")
            print("Pls make sure you have downloaded Pillow: pip install Pillow")
        
        plt.close()
    
    def create_pathfinding_gif(self, filename="pathfinding_demo.gif", fps=2):
        """创建最终Path的GIF"""
        
        assets_dir = "assets"
        if not os.path.exists(assets_dir):
            os.makedirs(assets_dir)
        
        full_path = os.path.join(assets_dir, filename)
        
        print(f"Creating pathfinding GIF: {full_path}")
        
        # 使用训练好的智能体走一遍迷宫
        temp_epsilon = self.epsilon
        self.epsilon = 0  # 不探索，只利用
        
        state = self.env.reset()
        path = [state]
        
        for step in range(100):  # 最多100步
            action = self.choose_action(state)
            next_state, reward, done = self.env.step(action)
            path.append(next_state)
            state = next_state
            
            if done:
                break
        
        self.epsilon = temp_epsilon  # 恢复原epsilon值
        
        # 创建动画
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 创建基础迷宫显示
        maze_display = self.env.get_maze_for_animation()
        cmap = plt.cm.colors.ListedColormap(['white', 'black', 'green', 'red'])
        im = ax.imshow(maze_display, cmap=cmap, interpolation='nearest')
        
        # 显示学习到的策略（红色箭头）
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if maze_display[row, col] == 0:  # 只在空地显示
                    state_key = (row, col)
                    if state_key in self.q_table and np.max(self.q_table[state_key]) > 0:
                        best_action = np.argmax(self.q_table[state_key])
                        dr, dc = self.env.actions[best_action]
                        ax.arrow(col, row, dc*0.3, dr*0.3, 
                               head_width=0.1, head_length=0.1, 
                               fc='red', ec='red', alpha=0.5)
        
        # 初始化智能体位置和路径
        agent_dot, = ax.plot([], [], 'bo', markersize=15)
        trail_line, = ax.plot([], [], 'b-', alpha=0.5, linewidth=2)
        
        title = ax.set_title('')
        ax.set_xticks(range(self.env.cols))
        ax.set_yticks(range(self.env.rows))
        ax.grid(True, alpha=0.3)
        
        def animate(frame):
            if frame < len(path):
                pos = path[frame]
                
                # 更新智能体位置
                agent_dot.set_data([pos[1]], [pos[0]])
                
                # 更新路径轨迹
                if frame > 0:
                    trail_x = [p[1] for p in path[:frame+1]]
                    trail_y = [p[0] for p in path[:frame+1]]
                    trail_line.set_data(trail_x, trail_y)
                
                # 更新标题
                if pos == self.env.goal_pos:
                    title.set_text(f'Pathfound！ {frame} Steps total')
                else:
                    title.set_text(f'Pathfinding... Steps: {frame}')
                
                return agent_dot, trail_line, title
            return agent_dot, trail_line, title
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=len(path),
                                     interval=1000//fps, blit=False, repeat=True)
        
        # 保存为GIF
        try:
            anim.save(full_path, writer='pillow', fps=fps)
            print(f"Pathfinding GIF saved as: {full_path}")
        except Exception as e:
            print(f"GIF saved failed: {e}")
            print("Pls make sure you have downloaded Pillow: pip install Pillow")
        
        plt.close()
        
        return len(path) - 1  # 返回步数