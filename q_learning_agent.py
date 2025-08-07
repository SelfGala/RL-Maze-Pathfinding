import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
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
        
        # Q表：使用字典存储状态-动作值
        self.q_table = defaultdict(lambda: np.zeros(len(env.actions)))
        
        # 训练统计
        self.episode_rewards = []
        self.episode_steps = []
        
        # 用于视频录制的数据
        self.training_positions = []  # 存储训练过程中的位置
        self.training_episodes = []   # 对应的episode数
    
    def choose_action(self, state):
        """epsilon-greedy策略选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, len(self.env.actions) - 1)  # 探索
        else:
            return np.argmax(self.q_table[state])  # 利用
    
    def update_q_value(self, state, action, reward, next_state):
        """更新Q值"""
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        
        # Q-learning更新公式
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    def train(self, episodes=1000, max_steps_per_episode=200, record_video=False, video_interval=50):
        """训练智能体"""
        print(f"开始训练，共{episodes}轮...")
        
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
                      f"平均奖励: {avg_reward:.2f}, "
                      f"平均步数: {avg_steps:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
    
    def test(self, num_tests=10):
        """测试训练后的智能体"""
        print(f"\n测试训练后的智能体...")
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
                    print(f"测试 {test + 1}: 成功！用了 {steps} 步")
                    break
            else:
                print(f"测试 {test + 1}: 失败（超时）")
        
        self.epsilon = test_epsilon  # 恢复原epsilon值
        
        if success_count > 0:
            avg_steps = total_steps / success_count
            print(f"\n测试结果: {success_count}/{num_tests} 成功")
            print(f"平均步数: {avg_steps:.1f}")
        else:
            print(f"\n测试结果: {success_count}/{num_tests} 成功")
    
    def plot_training_progress(self):
        """绘制训练进度"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制奖励曲线
        ax1.plot(self.episode_rewards)
        ax1.set_title('训练奖励')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True)
        
        # 绘制步数曲线
        ax2.plot(self.episode_steps)
        ax2.set_title('每轮步数')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def create_training_gif(self, filename="training_process.gif", fps=10):
        """创建训练过程的GIF动画"""
        if not self.training_positions:
            print("没有训练数据可用于创建GIF。请在训练时设置record_video=True")
            return
        
        print(f"正在创建训练过程GIF: {filename}")
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 创建基础迷宫显示
        maze_display = self.env.get_maze_for_animation()
        cmap = plt.cm.colors.ListedColormap(['white', 'black', 'green', 'red'])
        im = ax.imshow(maze_display, cmap=cmap, interpolation='nearest')
        
        # 初始化智能体位置
        agent_dot, = ax.plot([], [], 'bo', markersize=12)
        
        # 设置标题和网格
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
                
                # 更新标题
                title.set_text(f'训练过程 - Episode: {episode}, Step: {frame}')
                
                return agent_dot, title
            return agent_dot, title
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=len(self.training_positions),
                                     interval=1000//fps, blit=False, repeat=True)
        
        # 保存为GIF
        try:
            anim.save(filename, writer='pillow', fps=fps)
            print(f"训练过程GIF已保存为: {filename}")
        except Exception as e:
            print(f"保存GIF失败: {e}")
            print("请确保已安装Pillow: pip install Pillow")
        
        plt.close()
    
    def create_pathfinding_gif(self, filename="pathfinding_demo.gif", fps=2):
        """创建最终寻路演示的GIF"""
        print(f"正在创建寻路演示GIF: {filename}")
        
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
        
        # 设置标题和网格
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
                    title.set_text(f'寻路完成！总共用了 {frame} 步')
                else:
                    title.set_text(f'智能体寻路中... 步数: {frame}')
                
                return agent_dot, trail_line, title
            return agent_dot, trail_line, title
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=len(path),
                                     interval=1000//fps, blit=False, repeat=True)
        
        # 保存为GIF
        try:
            anim.save(filename, writer='pillow', fps=fps)
            print(f"寻路演示GIF已保存为: {filename}")
        except Exception as e:
            print(f"保存GIF失败: {e}")
            print("请确保已安装Pillow: pip install Pillow")
        
        plt.close()
        
        return len(path) - 1  # 返回步数