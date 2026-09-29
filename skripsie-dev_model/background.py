# Import Modules
import minigrid
import gymnasium as gym
#============================================================================

# Make the Gym Environment
env = gym.make('MiniGrid-Empty-8x8-v0', render_mode='human') 
    # 'human' allows us to see the rendered virtual environment
#============================================================================

# Agent takes Random Action
import random
a = random.randint(0, env.action_space.n -1)
print("Action a: %d was generated" % a)
#============================================================================

# Reset the Environment 
obs, info = env.reset()
print(obs)
print(info)
#============================================================================

# Extract Image Observation of Environment
from minigrid.wrappers import *
env = ImgObsWrapper(env)

obs, info = env.reset()
print(obs)
print(info)
#============================================================================

#============================================================================
#Extract Object_Idx Information
#============================================================================
# Method 1
def extractObjInfo(obs):
    (rows, cols, x) = obs.shape
    view = np.zeros((rows, cols))

    for r in range(rows):
        for c in range(cols):
            view[r, c] = obs[r, c, 0]

    return view

# Display
print("Method 1: Extract as a Matrix")
obs, _ = env.reset()
print(extractObjInfo(obs))
#============================================================================

# Method 2: Using Numpy Slicing and Reshaping
def extractObjInfo2(obs):
    (rows, cols, x) = obs.shape
    temp = np.reshape(obs, [rows*cols*x, 1], 'F')[0:rows*cols]
    return np.reshape(temp, [rows, cols], 'C')

#Display
print("Method 2: Using Numpy Slicing and Reshaping")
obs, _ = env.reset()
print(extractObjInfo2(obs))
#============================================================================

# Reward Signal
[o, r, d, t, i] = env.step(a)
print(o, r, d, t, i)

