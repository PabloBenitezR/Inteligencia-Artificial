import pygame
import numpy as np
import random

# Configuración de Pygame
pygame.init()
cell_size = 80
width, height = 4 * cell_size, 4 * cell_size
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Frozen Lake Q-Learning Agent")

# Cargar imágenes y escalarlas al tamaño de la celda
start_img = pygame.transform.scale(pygame.image.load("stool.png"), (cell_size, cell_size))
goal_img = pygame.transform.scale(pygame.image.load("goal.png"), (cell_size, cell_size))
hole_img = pygame.transform.scale(pygame.image.load("hole.png"), (cell_size, cell_size))
frozen_img = pygame.transform.scale(pygame.image.load("ice.png"), (cell_size, cell_size))
agent_img_up = pygame.transform.scale(pygame.image.load("elf_up.png"), (cell_size, cell_size))
agent_img_down = pygame.transform.scale(pygame.image.load("elf_down.png"), (cell_size, cell_size))
agent_img_left = pygame.transform.scale(pygame.image.load("elf_left.png"), (cell_size, cell_size))
agent_img_right = pygame.transform.scale(pygame.image.load("elf_right.png"), (cell_size, cell_size))



# Clase del entorno FrozenLake
class FrozenLake:
    def __init__(self, size=4):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.start_state = (0, 0)
        self.goal_state = (size-1, size-1)
        self.hole_states = [(1, 1), (1, 3), (2, 3), (3, 0)]
        for i, j in self.hole_states:
            self.grid[i][j] = 1

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        i, j = self.current_state
        if action == 0:  # mover hacia arriba
            i = max(i-1, 0)
        elif action == 1:  # mover hacia abajo
            i = min(i+1, self.size-1)
        elif action == 2:  # mover hacia la izquierda
            j = max(j-1, 0)
        elif action == 3:  # mover hacia la derecha
            j = min(j+1, self.size-1)
        
        self.current_state = (i, j)
        
        if self.current_state == self.goal_state:
            reward = 1
            done = True
        elif self.current_state in self.hole_states:
            reward = -1
            done = True
        else:
            reward = 0
            done = False
        
        return self.current_state, reward, done

    def render(self):
        for i in range(env.size):
            for j in range(env.size):
                x, y = j * cell_size, i * cell_size

                screen.blit(frozen_img, (x, y))

                if self.grid[i][j] == 1:
                    screen.blit(hole_img, (x, y))  # Agujero

                elif (i, j) == env.start_state:
                    screen.blit(start_img, (x, y))

                elif (i, j) == env.goal_state:
                    screen.blit(goal_img, (x, y))  # Meta

                if (i, j) == env.current_state:
                    screen.blit(agent_img_down, (x, y))  # Agente
                
        
        # Dibujar al agente
        agent_x, agent_y = self.current_state[1] * cell_size, self.current_state[0] * cell_size
        screen.blit(agent_img_down, (agent_x, agent_y))
        pygame.display.flip()

# Entrenamiento de Q-Learning
env = FrozenLake()
q_table = np.zeros((env.size, env.size, 4))
num_episodes = 10000
max_steps_per_episode = 100
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay_rate = 0.001

def epsilon_greedy_policy(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)
    else:
        return np.argmax(q_table[state[0]][state[1]])

for episode in range(num_episodes):
    state = env.reset()
    done = False
    t = 0
    while not done and t < max_steps_per_episode:
        action = epsilon_greedy_policy(state)
        next_state, reward, done = env.step(action)
        q_table[state[0]][state[1]][action] += learning_rate * \
            (reward + discount_factor * np.max(q_table[next_state[0]][next_state[1]]) - q_table[state[0]][state[1]][action])
        state = next_state
        t += 1
    epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay_rate))

# Ejecución de la política aprendida en pygame
def run_learned_policy(env, q_table):
    state = env.reset()
    done = False
    while not done:
        env.render()  # Muestra el entorno y la posición del agente
        pygame.time.delay(500)  # Espera medio segundo para ver el movimiento

        action = np.argmax(q_table[state[0]][state[1]])  # Mejor acción según Q-table
        next_state, reward, done = env.step(action)

        state = next_state

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        if done:
            if reward == 1:
                print("¡El agente alcanzó la meta!")
            else:
                print("El agente cayó en un agujero.")
            pygame.time.delay(1000)

run_learned_policy(env, q_table)
pygame.quit()
