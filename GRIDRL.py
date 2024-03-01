import numpy as np
obstacle_hit_reward = -10
class Gridworld:
    

    def __init__(self, width, height, start, goal, obstacles=None, rewards=None):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles if obstacles else []
        self.rewards = rewards if rewards else {}
        self.state = start
        self.actions = ['up', 'down', 'left', 'right']
    
    def move(self, action):
    
       

        if action == 'up':
            next_state = (max(self.state[0] - 1, 0), self.state[1])
        elif action == 'down':
            next_state = (min(self.state[0] + 1, self.height - 1), self.state[1])
        elif action == 'left':
            next_state = (self.state[0], max(self.state[1] - 1, 0))
        elif action == 'right':
            next_state = (self.state[0], min(self.state[1] + 1, self.width - 1))
        
        if next_state in self.obstacles:
            return self.state, obstacle_hit_reward
        
        # No obstacle
        self.state = next_state
        return self.state, self.rewards.get(self.state, 0)
    
    def is_goal_reached(self):
        return self.state == self.goal
    
    def reset(self):
        self.state = self.start
        return self.state

""" gridworld = Gridworld(width=5, height=5, start=(0, 0), goal=(4, 4), obstacles=[(1,2),(2, 2), (3,2), (4,1)], rewards={(4, 4): 10, (2, 2): -1})


def display_grid(state, width, height, goal, obstacles, clear_output=True):
    if clear_output:
        from IPython.display import clear_output
        clear_output(wait=True)
    
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    for obstacle in obstacles:
        grid[obstacle[0]][obstacle[1]] = 'X'  # Mark obstacles
    grid[goal[0]][goal[1]] = 'G'  # Mark goal
    grid[state[0]][state[1]] = 'A'  # Mark agent's current position
    for row in grid[::-1]:  # Print from top to bottom
        print(' '.join(row))
    print("\n")



start_state = gridworld.reset()
print("Start State: ", start_state)
display_grid(start_state, gridworld.width, gridworld.height, gridworld.goal, gridworld.obstacles)

# Test 
actions = ['right', 'left']
for action in actions:
    next_state, reward = gridworld.move(action)
    print(f"Action: {action}, Next State: {next_state}, Reward: {reward}")
    display_grid(next_state, gridworld.width, gridworld.height, gridworld.goal, gridworld.obstacles)
    if gridworld.is_goal_reached():
        print("Goal Reached!")
        break
 """




gridworld = Gridworld(
    width=8,
    height=8,
    start=(0, 0),
    goal=(7, 6),
    obstacles=[(2, 2), (3,2), (4,1)],
    rewards={(7,6): 10} 
)

class QLearningAgent:
    def __init__(self, alpha, gamma, epsilon, actions):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma  
        self.epsilon = epsilon  
        self.actions = actions 

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:  
            return np.random.choice(self.actions)
        else:  # Exploit
            q_values = [self.get_q_value(state, action) for action in self.actions]
            max_q = max(q_values)
            actions_with_max_q = [self.actions[i] for i in range(len(self.actions)) if q_values[i] == max_q]
            return np.random.choice(actions_with_max_q)

    def learn(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)
        max_next_q = max([self.get_q_value(next_state, a) for a in self.actions])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q


alpha = 0.4  
gamma = 0.99  
epsilon = 0.41  
agent = QLearningAgent(alpha, gamma, epsilon, gridworld.actions)



num_episodes = 1000  
max_steps_per_episode = 100 

rewards_all_episodes = []
# Training
for episode in range(num_episodes):
    state = gridworld.reset() 
    total_rewards = 0

    for step in range(max_steps_per_episode):
        action = agent.choose_action(state)  
        next_state, reward = gridworld.move(action)  

        if reward == obstacle_hit_reward:
            print(f"Episode {episode + 1} ended after {step + 1} steps due to obstacle.")
            

        agent.learn(state, action, reward, next_state) 
        state = next_state 
        total_rewards += reward 

        if gridworld.is_goal_reached(): 
            break

    rewards_all_episodes.append(total_rewards)  

rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000
print("Average reward per thousand episodes\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r)/1000))
    count += 1000

import pygame

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
GOLD = (255, 215, 0)
RED = (255, 0, 0)

CELL_SIZE = 60
GRID_WIDTH = gridworld.width * CELL_SIZE
GRID_HEIGHT = gridworld.height * CELL_SIZE

pygame.init()

def draw_grid(surface, obstacles, goal):
    for x in range(0, GRID_WIDTH, CELL_SIZE):
        for y in range(0, GRID_HEIGHT, CELL_SIZE):
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, WHITE, rect, 1)

    for obstacle in obstacles:
        rect = pygame.Rect(obstacle[1]*CELL_SIZE, (gridworld.height - 1 - obstacle[0])*CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, RED, rect)

    rect = pygame.Rect(goal[1]*CELL_SIZE, (gridworld.height - 1 - goal[0])*CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(surface, GOLD, rect)

def update_visualization(surface, state):
    surface.fill((0, 0, 0))

    draw_grid(surface, gridworld.obstacles, gridworld.goal)

    rect = pygame.Rect(state[1]*CELL_SIZE, (gridworld.height - 1 - state[0])*CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(surface, GREEN, rect)

    pygame.display.flip()

def run_simulation_with_visualization(gridworld, agent, num_episodes=100, visualize_every_n_episodes=20):
    surface = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT))
    pygame.display.set_caption('Grid RL')

    for episode in range(num_episodes):
        state = gridworld.reset()
        done = False

        for step in range(100):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.choose_action(state)
            next_state, reward = gridworld.move(action)
            agent.learn(state, action, reward, next_state)

            update_visualization(surface, state)

            pygame.time.delay(100)

            state = next_state
            done = gridworld.is_goal_reached()
            if done:
                break

        print(f"Episode: {episode + 1}, Steps taken: {step}, Reward: {reward}")

        pygame.time.delay(100)

    pygame.quit()

run_simulation_with_visualization(gridworld, agent, num_episodes=100, visualize_every_n_episodes=20)
