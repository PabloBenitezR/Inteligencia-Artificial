import pygame
import numpy as np
import random
import imageio  # Librería para crear GIF

# Configuración inicial de Pygame y variables de entorno
pygame.init()
cell_size = 80
width, height = 4 * cell_size, 4 * cell_size
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Frozen Lake Q-Learning Agent")

# Cargar imágenes de celdas y agente
start_img = pygame.transform.scale(pygame.image.load(r"C:\Users\ST\Desktop\Tareas Master\Inteligencia Artifical\Entregar IA-5\imagen\stool.png"), (cell_size, cell_size))
stool_img = pygame.transform.scale(pygame.image.load(r"C:\Users\ST\Desktop\Tareas Master\Inteligencia Artifical\Entregar IA-5\imagen\stool.png"), (cell_size, cell_size))
goal_img = pygame.transform.scale(pygame.image.load(r"C:\Users\ST\Desktop\Tareas Master\Inteligencia Artifical\Entregar IA-5\imagen\goal.png"), (cell_size, cell_size))
hole_img = pygame.transform.scale(pygame.image.load(r"C:\Users\ST\Desktop\Tareas Master\Inteligencia Artifical\Entregar IA-5\imagen\hole.png"), (cell_size, cell_size))
frozen_img = pygame.transform.scale(pygame.image.load(r"C:\Users\ST\Desktop\Tareas Master\Inteligencia Artifical\Entregar IA-5\imagen\ice.png"), (cell_size, cell_size))
agent_img_up = pygame.transform.scale(pygame.image.load(r"C:\Users\ST\Desktop\Tareas Master\Inteligencia Artifical\Entregar IA-5\imagen\elf_up.png"), (cell_size, cell_size))
agent_img_down = pygame.transform.scale(pygame.image.load(r"C:\Users\ST\Desktop\Tareas Master\Inteligencia Artifical\Entregar IA-5\imagen\elf_down.png"), (cell_size, cell_size))
agent_img_left = pygame.transform.scale(pygame.image.load(r"C:\Users\ST\Desktop\Tareas Master\Inteligencia Artifical\Entregar IA-5\imagen\elf_left.png"), (cell_size, cell_size))
agent_img_right = pygame.transform.scale(pygame.image.load(r"C:\Users\ST\Desktop\Tareas Master\Inteligencia Artifical\Entregar IA-5\imagen\elf_right.png"), (cell_size, cell_size))

# Definición de la clase de entorno FrozenLake
class FrozenLake:
    def __init__(self, size=4):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.start_state = (0, 0)
        self.goal_state = (size-1, size-1)
        self.hole_states = [(1, 1), (1,3), (2, 3), (3, 0)]
        for i, j in self.hole_states:
            self.grid[i][j] = 1

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        i, j = self.current_state
        if action == 0:  # Arriba
            i = max(i-1, 0)
        elif action == 1:  # Abajo
            i = min(i+1, self.size-1)
        elif action == 2:  # Izquierda
            j = max(j-1, 0)
        elif action == 3:  # Derecha
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
        screen.fill((255, 255, 255))  # Fondo blanco
        for i in range(env.size):
            for j in range(env.size):
                x, y = j * cell_size, i * cell_size

                screen.blit(frozen_img, (x, y))

                if self.grid[i][j] == 1:
                    screen.blit(hole_img, (x, y))  # Agujero

                elif (i, j) == env.start_state:
                    screen.blit(stool_img, (x, y))

                elif (i, j) == env.goal_state:
                    screen.blit(goal_img, (x, y))  # Meta

                if (i, j) == env.current_state:
                    screen.blit(agent_img_down, (x, y))  # Agente
        
        # Dibujar el agente
        agent_x, agent_y = self.current_state[1] * cell_size, self.current_state[0] * cell_size
        screen.blit(agent_img_down, (agent_x, agent_y))
        pygame.display.flip()

# Entrenamiento Q-Learning
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

# Captura de fotogramas
frames = []

def run_learned_policy(env, q_table):
    state = env.reset()
    done = False
    while not done:
        env.render()
        
        # Capturar la pantalla y guardarla como imagen para el GIF
        image_data = pygame.surfarray.array3d(screen)
        frames.append(image_data)
        
        pygame.time.delay(300)  # Ajustar la velocidad del movimiento

        action = np.argmax(q_table[state[0]][state[1]])  # Selección de la mejor acción
        next_state, reward, done = env.step(action)

        state = next_state

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        if done:
            if reward == 1:
                print("¡El agente alcanzó la meta!")
                
                # Mantener en el goal durante 5 segundos
                for _ in range(50):  # 50 iteraciones a 100 ms cada una
                    env.render()
                    image_data = pygame.surfarray.array3d(screen)
                    frames.append(image_data)
                    pygame.time.delay(100)  # 0.1 segundos = 100 ms

            else:
                print("El agente cayó en un agujero.")
            pygame.time.delay(1000)

run_learned_policy(env, q_table)

# Crear el GIF con imageio
with imageio.get_writer("frozen_lake_simulation.gif", mode="I", duration=0.3) as writer:
    for frame in frames:
        writer.append_data(frame)

pygame.quit()

