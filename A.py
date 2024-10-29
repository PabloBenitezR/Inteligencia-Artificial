import numpy as np
import random


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
        if action == 0:  #  arriba
            i = max(i-1, 0)
        elif action == 1:  #  abajo
            i = min(i+1, self.size-1)
        elif action == 2:  # izquierda
            j = max(j-1, 0)
        elif action == 3:  #  derecha
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
        print('\n')
        for i in range(self.size):
            for j in range(self.size):
               


                if self.grid[i][j] == 0:

                    if (i, j) == self.current_state: #agente
                        print('A', end=' ')  
                    elif (i, j) == self.goal_state: #meta
                        print('G', end=' ')  
                    else:
                        print('.', end=' ')    #hielo
                elif self.grid[i][j] == 1:
                    if (i, j) == self.current_state:
                        print('A', end=' ')  #agente
                    else:
                        print('X', end=' ')  #hoyo
            print()
        print()
    
    # Q-tabla
    def show_q_tabla(self, q_tabla):
        print('-----------------------------------------------------------------')
        print('Q-Table:')
        print('-----------------------------------------------------------------')
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == 0:
                    print( '%.2f' % q_tabla[i][j][0], end='\t')
                    print('%.2f' % q_tabla[i][j][1], end='\t')
                    print('%.2f' % q_tabla[i][j][2], end='\t')
                    print('%.2f' % q_tabla[i][j][3])
                else:
                    print('NULL', end='\t')
                    print('NULL', end='\t')
                    print('NULL', end='\t')
                    print('NULL')
            print()

# Ambiente
env = FrozenLake()

# Q-tabla con 0's
q_tabla = np.zeros((env.size, env.size, 4))

# hiperparámetros
num_episodes = 10000
max_steps_per_episode = 100
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay_rate = 0.001

# política
def egreedy_poli(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)
    else:
        return np.argmax(q_tabla[state[0]][state[1]])

# Entrenamiento del agente
for episode in range(num_episodes):
    state = env.reset()
    done = False
    t = 0
    while not done and t < max_steps_per_episode:
        action = egreedy_poli(state)
        next_state, reward, done = env.step(action)
        q_tabla[state[0]][state[1]][action] += learning_rate * \
            (reward + discount_factor * np.max(q_tabla[next_state[0]][next_state[1]]) - q_tabla[state[0]][state[1]][action])
        state = next_state
        t += 1
    epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay_rate))

    # Ver el progreso cada 1000 episodios para contar un total de 10000 episodios
    if episode % 1000 == 0:
        print(f"Episodio {episode}")
        env.render()
        env.show_q_tabla(q_tabla)

# Después el agente corre lo aprendido con la política de selección ya entrenada
def run_learned_policy(env, q_tabla):
    state = env.reset() 
    done = False
    steps = 0

    print("\nEl agente está corriendo lo aprendido..\n")
    env.render()  

    while not done:
        action = np.argmax(q_tabla[state[0]][state[1]])  # escogiendo el mejor valor de la Q-tabla
        next_state, reward, done = env.step(action)

        
        env.render()

       
        print(f"Paso {steps + 1}: Estado {state} -> Acción {['UP', 'DOWN', 'LEFT', 'RIGHT'][action]}")

        
        state = next_state
        steps += 1

        if done:
            if reward == 1:
                print("\n¡El agente llegó al regalo!")
            else:
                print("\n¡El agente se cayó en un hoyo!")
            break


run_learned_policy(env, q_tabla)
