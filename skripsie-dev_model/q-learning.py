#============================================================================
# Import Modules
#============================================================================
# Imports for Environment
import minigrid
import gymnasium as gym
from minigrid.wrappers import *

# Imports for Generate Random Numbers / Epsilon Decay
import random
import math
# Import for Performance Tracking
import time

# Import for Various Mathematical, Vector and Matrix Functions
import numpy as np

# Import for Hashing   
import hashlib
# Import for Hyperparameter Tuning
import optuna

# Import for File Handling
import pickle
from os.path import exists

# Imports for Writing to Tensorboard
from torch.utils.tensorboard import SummaryWriter

#============================================================================
# Define Functions
#============================================================================
# Extract Object_Idx Info Using Numpy Slicing and Reshaping
def extractObjInfo(obs):
    (rows, cols, x) = obs.shape
    temp = np.reshape(obs, [rows*cols*x, 1], 'F')[0:rows*cols]
    return np.reshape(temp, [rows, cols], 'C')

# Generate a Unique Key for Each State
def hashState(state):
    # Convert the Numpy Array to Bytes, then Hash it with MD5
    stateBytes = state.tobytes()
    hashValue = int(hashlib.md5(stateBytes).hexdigest(), 16)
    return hashValue

# ---- LOAD Q-TABLE IF IT EXISTS ----
def load_q_table(filename):
    if (exists(filename)):
        print('Loading Existing Q Values...')
        # Load Data (Deserialise)
        with open(filename, 'rb') as handle:
            return pickle.load(handle)
            handle.close()
    else:
        print('Filename %s DNE: Could Not Load Data' % filename)
        return {}

#============================================================================
# Configuration
#============================================================================
# Establish Environment Parameters
n_actions = 3
max_steps = gym.make('MiniGrid-Empty-8x8-v0').unwrapped.max_steps

# Variable for storing the Tabular Value-Function
filename = 'qtable.pickle'

#============================================================================
# Q-Table Training Function
#============================================================================
def train_q_learning(env, alpha, gamma, epsilon_start, epsilon_decay, epsilon_min, 
                     episodes = 3000, log_to_tb = False, writer = None, initial_q = None):
    # SetUp Counter
    total_steps = 0
    min_steps = 1000

    # Definte Q Table
    q_table = dict(initial_q) if initial_q is not None else {}
    
    # Define Training Variables
    epsilon = epsilon_start
    episode_rewards = []

    # Episode Loop
    for e in range(episodes):
        # Reset the Environment
        obs, _ = env.reset()

        # Extract Current State
        state = extractObjInfo(obs)

        # State Hash Value
        stateKey = hashState(state)
        if stateKey not in q_table: # prevent KeyError on Unseen States
            q_table[stateKey] = np.zeros(n_actions)*1.0
        
        # Declare Tracking Variables
        done = False
        loss = 0.0
        total_reward = 0
        episode_steps =0
    
        # Agent Step Loop
        for s in range(0, max_steps):
            #============================================================================
            # Epsilon-Greedy Exploration
            #============================================================================
            # Perform Epsilon Greedy Action
            if (random.random() < epsilon):
                # Explore Environment - Select Random Action
                a = random.randint(0, n_actions-1)
            else:
                # Exploit Environment - Select Action for max of Value Function @ Current State
                a = np.argmax(q_table[stateKey])

            # Extract Step Information
            obs, reward, done, truncated, _ = env.step(a)

            # Extract Next State from Observation
            state2 = extractObjInfo(obs)

            # Hash the Next State
            state2Key = hashState(state2)
            if state2Key not in q_table:
                q_table[state2Key] = np.zeros(n_actions)

            #============================================================================
            # ---- Q-TABLE UPDATE (Bellman Equation) ----
            #============================================================================
            # Q-Learning
            error = reward + gamma*np.max(q_table[state2Key]) - q_table[stateKey][a]
            loss += error**2
            episode_steps += 1
            q_table[stateKey][a] = q_table[stateKey][a] + alpha*error


            # Increment Count Every Step
            total_steps += 1
            # Calculate Accumulated Rewards
            total_reward += reward

            # Render the Environment
            #env.render()

            # Move to Next State
            state = state2
            stateKey = state2Key
            

            # Goal was/wasn't Reached
            if (done or truncated):      
                break

        # Decay Epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Epsiode Rewards
        episode_rewards.append(total_reward)

        # Track the Minimum Steps Taken
        if (min_steps > episode_steps):
            min_steps = episode_steps    
        

        # Write to Tensorboard upon Completion
        if log_to_tb and writer is not None:
            writer.add_scalar("Reward/train", total_reward, total_steps)
            writer.add_scalar("Loss/train", loss/episode_steps, total_steps)
            writer.add_scalar("Epsilon/train", epsilon, total_steps)
            writer.add_scalar("Steps/train", episode_steps, total_steps)

    return np.mean(episode_rewards[-100:]), q_table, min_steps

#============================================================================
# Tune Hyperparameters using Optuna
#============================================================================
def objective(trial):
    # Tune Model Hyperparameters
    alpha = trial.suggest_float("alpha", 0.01, 1.0, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    epsilon_start = trial.suggest_float("epsilon_start", 0.5, 1.0)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.99, 0.9999)
    epsilon_min = trial.suggest_float("epsilon_min", 0.001, 0.1)

    # Declare Empty Rewards Array
    rewards = []
    # Start Tuning Time
    start_time = time.time()

    for seed in range(3): 
        np.random.seed(seed)
        random.seed(seed)

        # Make the Gym Environment
        env = gym.make('MiniGrid-Empty-8x8-v0', render_mode=None).unwrapped
        env = ImgObsWrapper(env) 

        # Search the Algorithm
        mean_reward,_,_ = train_q_learning(
            env, alpha, gamma, epsilon_start, epsilon_decay, epsilon_min,
            episodes=800, log_to_tb=False, writer=None, initial_q=None) 
        
        rewards.append(mean_reward)
        env.close()

    
    elapsed = time.time() - start_time
    print(f'Trial Done In {elapsed:.2f} s ({elapsed/60:.2f} min)')

    # Return the Average of the Trial Rewards
    return np.mean(rewards)


#============================================================================
# Run the Study
#============================================================================
if __name__ == "__main__":
    # Check for Existing Q-Table
    existing_q = load_q_table(filename)

    # Study the Model
    print("Start Hyperparameter Tuning")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials= 50)

    # Display the Best Parameters
    print("Best Params:", study.best_params)
    print("Best Value:", study.best_value)

    # SetUp Tensorboard
    writer = SummaryWriter()
    # Extract the Best Parameters
    best = study.best_params

    # Make the Gym Environment
    env = gym.make('MiniGrid-Empty-8x8-v0', render_mode=None).unwrapped
    env = ImgObsWrapper(env) 
    
    # Start Training
    print('Start Training...')
    # Start Training Time
    start_time = time.time()

    # Train the Model 
    _, trained_qtable, final_steps = train_q_learning(
        env,
        alpha = best["alpha"],
        gamma = best["gamma"],
        epsilon_start = best["epsilon_start"],
        epsilon_decay = best["epsilon_decay"], 
        epsilon_min = best["epsilon_min"],
        episodes = 3000,
        log_to_tb = True,
        writer = writer,
        initial_q=existing_q
    )

    # Close the Environment
    env.close()

    # Done Training
    print('Done Training...')
    end_time = time.time()
    elapsed = end_time - start_time
    print(f'Training Done In {elapsed:.2f} s ({elapsed/60:.2f} min)')
    print(f'Final Steps Taken: {final_steps}')

    # Flush Remaining Data
    writer.flush()
    writer.close()

    # ---- SAVE Q-TABLE AFTER TRAINING ----
    with open(filename, 'wb') as handle:
        pickle.dump(trained_qtable, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
