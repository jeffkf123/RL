import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64) 
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

class Gridworld:
    def __init__(self, width, height, start, goal, obstacles=None, rewards=None):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles if obstacles else [])
        self.rewards = rewards if rewards else {}
        self.actions = ['up', 'down', 'left', 'right']
        self.state = self.start  

    def _encode_state(self, position):
        """One-hot encode the grid position."""
        encoded = np.zeros(self.width * self.height, dtype=np.float32)
        encoded[position[0] * self.width + position[1]] = 1.0
        return encoded

    def move(self, action):
        """Move to the next state based on action."""
        # Define movement logic
        if action == 'up':
            next_state = (max(self.state[0] - 1, 0), self.state[1])
        elif action == 'down':
            next_state = (min(self.state[0] + 1, self.height - 1), self.state[1])
        elif action == 'left':
            next_state = (self.state[0], max(self.state[1] - 1, 0))
        elif action == 'right':
            next_state = (self.state[0], min(self.state[1] + 1, self.width - 1))
        else:
            next_state = self.state

      
        if next_state in self.obstacles:
            reward = -10  
            done = True  
        elif next_state == self.goal:
            reward = self.rewards.get(next_state, 0)
            done = True  
        else:
            reward = self.rewards.get(next_state, 0)  
            done = False

      
        self.state = next_state

      
        return self._encode_state(next_state), reward, done

    def reset(self):
        """Reset the environment to the start state."""
        self.state = self.start
        return self._encode_state(self.start)
import random
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Randomly sample a batch of transitions from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma  # discount
        self.epsilon = epsilon  # exploration
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def remember(self, state, action, reward, next_state, done):
        # Remember an experience
        self.memory.push(torch.FloatTensor([state]), torch.LongTensor([[action]]), torch.FloatTensor([reward]), torch.FloatTensor([next_state]), done)

    def choose_action(self, state):
        #Choose an action based on epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = self.model(state)
            return np.argmax(action_values.cpu().data.numpy())

    def replay(self):
        #Experience replay to update the neural network 
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.model(state_batch).gather(1, action_batch)

        #V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()
        # expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_epsilon(self):
        #exponential decay
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)


gridworld = Gridworld(
    width=7,  
    height=7,  
    start=(0, 0), 
    goal=(6 ,6), 
    obstacles=[(3, 1), (4, 2), (5, 3)],  
    rewards={(6, 6): 300}  
)


state_size = gridworld.width * gridworld.height  
action_size = 4  

agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    learning_rate=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    memory_size=10000,
    batch_size=64
)

num_episodes = 1000  
max_steps_per_episode = 100  
update_every = 4  
goal_reached = 0  

for episode in range(num_episodes):

    state = gridworld.reset()
    total_reward = 0
    done = False

    for step in range(max_steps_per_episode):
   
        action = agent.choose_action(state)

        actions = ['up', 'down', 'left', 'right']
        next_state, reward, done = gridworld.move(gridworld.actions[action])

  
        agent.remember(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

      
        if done:
            if reward > 0:  # Assuming positive reward is only from reaching the goal
                goal_reached += 1
            break

        
        if step % update_every == 0:
            agent.replay()

   
    agent.update_epsilon()

    if episode % 100 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}, Goal Reached: {goal_reached}")

import pygame
import time


pygame.init()


WIDTH, HEIGHT = 500, 500
GRID_SIZE = 7
CELL_SIZE = WIDTH // GRID_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gridworld")

def draw_grid():
    for x in range(0, WIDTH, CELL_SIZE):
        for y in range(0, HEIGHT, CELL_SIZE):
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, WHITE, rect, 1)

def draw_environment(environment):
    for obstacle in environment.obstacles:
        pygame.draw.rect(screen, WHITE, (obstacle[1]*CELL_SIZE, obstacle[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, GREEN, (environment.goal[1]*CELL_SIZE, environment.goal[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, RED, (environment.start[1]*CELL_SIZE, environment.start[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

def update_agent_position(position):
    pygame.draw.rect(screen, BLUE, (position[1]*CELL_SIZE, position[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

def visualize_gridworld(agent, environment, steps=100):
    running = True
    step = 0
    while running and step < steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))
        draw_grid()
        draw_environment(environment)
        update_agent_position(environment.state)

        pygame.display.flip()

        action = agent.choose_action(environment._encode_state(environment.state))
        _, _, done = environment.move(environment.actions[action])

        if done:
            environment.reset()
            agent.update_epsilon()

        step += 1
        time.sleep(0.1)

    pygame.quit()


visualize_gridworld(agent, gridworld)
