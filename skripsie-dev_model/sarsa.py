#============================================================================
# Import Modules
#============================================================================
import minigrid
import gymnasium as gym
from minigrid.wrappers import *

import random
import time
import pickle
from os.path import exists

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

import hashlib

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

#============================================================================
# SetUp the RL Agent
#============================================================================

# Make the Gym Environment
env = gym.make('MiniGrid-Empty-8x8-v0', render_mode=None).unwrapped
    # 'human' allows us to see the rendered virtual environment

# Variable for storing the Tabular Value-Function
Q = {}
filename = 'sarsa.pickle'

# ---- LOAD Q-TABLE IF IT EXISTS ----
if (exists(filename)):
    print('Loading Existing Q Values...')
    # Load Data (Deserialise)
    with open(filename, 'rb') as handle:
        Q = pickle.load(handle)
        handle.close()
else:
    print('Filename %s DNE: Could Not Load Data' % filename)

# Ranges
numActions = 3 # first 3 actions
episodes = 3000
maxSteps = env.max_steps

# Wrapper - Observation will only contain Grid Information
env = ImgObsWrapper(env)

# Reset the Environment
obs, _ = env.reset()

# Extract Current State
state = extractObjInfo(obs)

'''initialise the initial values of the value-function to be zero 
    - this is a pessimistic initialisation
## note that using the numpy array of the observation will not work in practice, 
    you will need to calculate a hash-value of the current state and 
    use it as unique key into the dictionary '''

# State Hash Value
stateKey = hashState(state)
if stateKey not in Q: # prevent KeyError on Unseen States
    Q[stateKey] = np.zeros(numActions)

# Training Variables
epsilon = 0.99
epsilon_decay = 0.99995
epsilon_min = 0.01

alpha = 0.1   # learning rate
gamma = 0.99  # discount factor

# Plotting SetUp
steps_done = 0

#============================================================================
# Main RL Loop  (Max Steps: 256)
#============================================================================
# Start Training
print('Start Training...')
start_time = time.time()

# Episode Loop
for e in range(episodes):
    # Reset the Environment
    obs, _ = env.reset()

    # Extract Current State
    state = extractObjInfo(obs)

    # State Hash Value
    stateKey = hashState(state)
    if stateKey not in Q: # prevent KeyError on Unseen States
        Q[stateKey] = np.zeros(numActions)*1.0

    #============================================================================
    # SARSA Requirements
    #============================================================================
    # Select First Action 
    if random.random() < epsilon:
        a = random.randint(0, numActions-1)
    else:
        a = np.argmax(Q[stateKey])
   
    # Agent Step Loop
    for s in range(0, maxSteps):

        # Agent takes Random Action
        #//a = random.randint(0, numActions)
        # Extract Step Information
        obs, reward, done, truncated, info = env.step(a)

        #============================================================================
        # Epsilon-Greedy Exploration
        #============================================================================
        # Extract Next State from Observation
        state2 = extractObjInfo(obs)

       # Hash the Next State
        state2Key = hashState(state2)
        if state2Key not in Q:
            Q[state2Key] = np.zeros(numActions)
                        
        # Select Next Action Using Epsilon-Greedy
        if random.random() < epsilon:
            a2 = random.randint(0, numActions-1)
        else:
            a2 = np.argmax(Q[state2Key])

        ''' take action 'a', receive reward 'reward', and observe next state 'obs'
        'done' boolean variable that indicates if the termination state was reached
        'truncated' boolean variable indicates if episode ended before reaching termination state
        'info' information provided by the gym environment '''


        #============================================================================
        # ---- Q_TABLE UPDATE (Bellman Equation) ----
        #============================================================================
        # SARSA Update
        error = reward + gamma * Q[state2Key][a2] - Q[stateKey][a]
        loss = error ** 2
        Q[stateKey][a] = Q[stateKey][a] + alpha * error
        
        # Decay Epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Increment Count Every Step
        steps_done += 1

        # Render the Environment
        #env.render()

        # Move to Next State
        state = state2
        stateKey = state2Key
        a = a2

        # Goal was Reached
        if (done == True):      
            print('Episode %d Finished Successfully:' % e)
            print('Steps: %d' % s)
            print('Reward: %f' % reward)
            break

        # No More Steps Allowed
        if (truncated == True):
            print('Episode %d Finished Unsuccessful - Truncated:' % e)
            print('Steps: %d' % s)
            print('Reward: %f' % reward)
            break

    # Write to Tensorboard
    if done:
        writer.add_scalar("Reward/train", reward, steps_done)
        writer.add_scalar("Loss/train", loss, steps_done)
        writer.add_scalar("Epsilon/train", epsilon, steps_done)

# Done Training
print('Done Training...')
end_time = time.time()
elapsed = end_time - start_time
print(f'Training took {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)')

# Flush Remaining Data
writer.flush()
writer.close()

# ---- SAVE Q-TABLE AFTER TRAINING ----
with open(filename, 'wb') as handle:
    pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()