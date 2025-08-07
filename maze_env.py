import numpy as np
import matplotlib.pyplot as plt
import random

class MazeEnvironment:
    def __init__(self, maze_size=(10, 10), use_random=True, wall_probability=0.3):
        """
        初始化迷宫环境
        0: 空地 (可通行)
        1: 墙壁 (不可通行)
        2: 起点
        3: 终点
        
        Args:
            maze_size: 迷宫大小 (rows, cols)
            use_random: 是否随机生成迷宫
            wall_probability: 随机生成时墙壁的概率
        """
        self.rows, self.cols = maze_size
        self.maze = self.generate_maze(use_random, wall_probability)
        self.start_pos = self.find_position(2)
        self.goal_pos = self.find_position(3)
        self.current_pos = self.start_pos
        
        # 动作空间：上、下、左、右
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
    def generate_maze(self, use_random=True, wall_probability=0.3):
        """生成迷宫
        
        Args:
            use_random: 是否使用随机生成，False则使用固定布局
            wall_probability: 随机生成时墙壁的概率 (0.0-1.0)
        """
        if not use_random:
            return self._generate_fixed_maze()
        
        # 随机生成迷宫
        maze = np.zeros((self.rows, self.cols))
        
        # 随机放置墙壁
        for row in range(self.rows):
            for col in range(self.cols):
                if random.random() < wall_probability:
                    maze[row, col] = 1
        
        # 设置起点和终点
        start_row, start_col = 0, 0
        goal_row, goal_col = self.rows - 1, self.cols - 1
        
        # 确保起点和终点不是墙壁
        maze[start_row, start_col] = 2  # 起点
        maze[goal_row, goal_col] = 3    # 终点
        
        # 确保起点和终点周围至少有一条通路
        self._ensure_path_around_position(maze, start_row, start_col)
        self._ensure_path_around_position(maze, goal_row, goal_col)
        
        # 检查是否存在从起点到终点的路径，如果没有则重新生成
        if not self._has_path(maze, (start_row, start_col), (goal_row, goal_col)):
            print("生成的迷宫无解，重新生成...")
            return self.generate_maze(use_random, wall_probability * 0.8)  # 减少墙壁密度重试
        
        return maze
    
    def _generate_fixed_maze(self):
        """生成固定布局的迷宫（原版本）"""
        maze = np.zeros((self.rows, self.cols))
        
        # 添加一些墙壁
        maze[1:3, 2:5] = 1
        maze[4:6, 1:3] = 1
        maze[3:5, 6:8] = 1
        maze[7:9, 3:6] = 1
        maze[6:8, 7:9] = 1
        
        # 设置起点和终点
        maze[0, 0] = 2  # 起点
        maze[-1, -1] = 3  # 终点
        
        return maze
    
    def _ensure_path_around_position(self, maze, row, col):
        """确保指定位置周围有通路"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.rows and 
                0 <= new_col < self.cols):
                if maze[new_row, new_col] == 1:  # 如果是墙壁，改为通路
                    maze[new_row, new_col] = 0
                    break
    
    def _has_path(self, maze, start, goal):
        """使用BFS检查是否存在从起点到终点的路径"""
        from collections import deque
        
        queue = deque([start])
        visited = set([start])
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            row, col = queue.popleft()
            
            if (row, col) == goal:
                return True
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < self.rows and 
                    0 <= new_col < self.cols and 
                    (new_row, new_col) not in visited and 
                    maze[new_row, new_col] != 1):  # 不是墙壁
                    
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col))
        
        return False  # 没找到路径
    
    def find_position(self, value):
        """找到指定值在迷宫中的位置"""
        pos = np.where(self.maze == value)
        return (pos[0][0], pos[1][0])
    
    def is_valid_position(self, pos):
        """检查位置是否有效（在边界内且不是墙壁）"""
        row, col = pos
        return (0 <= row < self.rows and 
                0 <= col < self.cols and 
                self.maze[row, col] != 1)
    
    def reset(self):
        """重置环境到初始状态"""
        self.current_pos = self.start_pos
        return self.current_pos
    
    def step(self, action):
        """执行动作，返回新状态、奖励、是否结束"""
        if action < 0 or action >= len(self.actions):
            return self.current_pos, -10, False  # 无效动作
        
        # 计算新位置
        dr, dc = self.actions[action]
        new_pos = (self.current_pos[0] + dr, self.current_pos[1] + dc)
        
        # 检查新位置是否有效
        if not self.is_valid_position(new_pos):
            return self.current_pos, -1, False  # 撞墙惩罚
        
        # 更新当前位置
        self.current_pos = new_pos
        
        # 计算奖励
        if self.current_pos == self.goal_pos:
            return self.current_pos, 100, True  # 到达目标
        else:
            return self.current_pos, -0.1, False  # 每步小惩罚
    
    def render(self, q_table=None, show_values=False, ax=None, title=""):
        """可视化迷宫"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        ax.clear()
        
        # 绘制迷宫
        display_maze = self.maze.copy()
        
        # 设置颜色映射
        cmap = plt.cm.colors.ListedColormap(['white', 'black', 'green', 'red'])
        ax.imshow(display_maze, cmap=cmap, interpolation='nearest')
        
        # 如果有Q表，显示最佳动作
        if q_table is not None and show_values:
            for row in range(self.rows):
                for col in range(self.cols):
                    if display_maze[row, col] == 0:  # 只在空地显示
                        state = (row, col)
                        if state in q_table and np.max(q_table[state]) > 0:
                            best_action = np.argmax(q_table[state])
                            # 绘制箭头表示最佳动作
                            dr, dc = self.actions[best_action]
                            ax.arrow(col, row, dc*0.3, dr*0.3, 
                                   head_width=0.1, head_length=0.1, 
                                   fc='red', ec='red', alpha=0.7)
        
        # 标记当前位置
        ax.plot(self.current_pos[1], self.current_pos[0], 'bo', markersize=12)
        
        # 设置网格
        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        ax.grid(True, alpha=0.3)
        ax.set_title(title or 'Maze - Blue: Agent, Green: Start, Red: Goal, Black: Wall')
        
        if ax is None:
            plt.show()
    
    def get_maze_for_animation(self):
        """返回用于动画的迷宫数据"""
        return self.maze.copy()