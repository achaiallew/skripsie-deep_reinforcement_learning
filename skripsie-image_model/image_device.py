#============================================================================
# Import Modules
#============================================================================
# Imports for Environment
import gymnasium as gym
import minigrid
from minigrid.wrappers import *

# Import for Performance Tracking
import time

# Import for Embedded Device Communication
import serial
# Import for Various Mathematical, Vector and Matrix Functions
import numpy as np

# Import for Experience Replay Memory
from collections import namedtuple, deque

# Imports for Neural Network
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Check for GPU Availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#============================================================================
# Preprocessing Functions - RGB Image
#============================================================================
# RGB Image parameters
screen_height = 56
screen_width = 56

# Compute Grayscale Version of RGB Frame
def grayscale(frame):
    # Grayscale Frame
    return np.mean(frame,-1)

# Normalize Pixel Values
def normalize_frame(frame, max_value):
    return frame/max_value

# Combine Preprocessing Fuctions
def preprocess_frame(frame):
    return normalize_frame(grayscale(frame), 255.0)
# This will give a frame of shape [screen_height, screen_width] having values between [0,1]

#============================================================================
# Stack the Frames - RGB Image
#============================================================================
# Frame Stack Size (4 Frames)
stack_size = 4

''' We return both the stacked frames as a single tensor object, and the deque that contains the individual frames
    The boolean 'is_new_episode' is used to indicate that the input frame must be fed to the deque 'stack_size' times'''

# Stack Frames
def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)
    
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((screen_height,screen_width), dtype=np.int32) for i in range(stack_size)], maxlen=stack_size)
        
        for i in range(stack_size):
            # Because we're in a new episode, copy the same frame 'stack_size' times
            stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = torch.from_numpy(np.stack(stacked_frames, axis=0)).float().unsqueeze(0)
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = torch.from_numpy(np.stack(stacked_frames, axis=0)).float().unsqueeze(0)
    
    return stacked_state, stacked_frames

#============================================================================
# Configuration
#============================================================================
# Configure Max Steps
max_steps = gym.make('MiniGrid-Empty-8x8-v0', render_mode=None).unwrapped.max_steps

# Configure Serial Communication
pico = serial.Serial("COM3", 115200, timeout=5)

#============================================================================
# Raspberry Pi Pico Communication
#============================================================================
def get_action(state):
    # Set State as Numpy Array
    data = (state.squeeze(0).numpy() * 255).round().clip(0, 255).astype(np.uint8)

    # Reset Input Buffer
    pico.reset_input_buffer()

    # Send State Image
    pico.write(data)

    # Wait for Answer
    reply = pico.readline()

    # Return Action
    return int(reply.decode().strip())

#============================================================================
# Evaluate Agent Performance
#============================================================================
def eval_model(env, eval_episodes=100, print_results=True):
    # Set Up Eval Counter Variables
    eval_success = 0.0
    total_steps = 0.0
    total_reward = 0.0

    # Episode Loop
    with torch.no_grad():
        for e in range(eval_episodes):
            # Reset the Environment
            current_obs, _ = env.reset()
            # Initialize the Frame Stack
            stacked_frames = deque([np.zeros((screen_height,screen_width), dtype=np.int32) 
                                    for i in range(stack_size)], maxlen=stack_size)
            # Preprocess the Observation to Obtain State
            current_state, stacked_frames = stack_frames(stacked_frames, current_obs, True)
            # Main RL Loop
            for s in range(0, max_steps):

                # Receive the Action
                action = get_action(current_state)

                # Take Action
                next_obs, reward, done, truncated, info = env.step(action)

                # Observe a New State
                if done or truncated:
                    next_state = None
                else:
                    next_state, stacked_frames = stack_frames(stacked_frames, next_obs, False)

                # Calculate Reward
                if done or truncated:
                    total_reward += reward
                    total_steps += env.unwrapped.step_count

                    # Tally Evaluation Successes
                    if done:
                        eval_success += 1

                    # Display Results
                    if print_results:
                        result = "reached goal" if done else "failed"
                        print(f"Episode {e}: reward={reward:.4f}, "
                            f"steps={env.unwrapped.step_count}, {result}")
                        
                    break

                # Move to the Next State
                current_obs = next_obs.copy()
                current_state = next_state

    success_rate = 100 * eval_success / eval_episodes
    mean_reward = total_reward / eval_episodes
    mean_steps = total_steps / eval_episodes

    return mean_reward, success_rate, mean_steps


#============================================================================
# Run the Study
#============================================================================
if __name__ == "__main__":
    # Gym Environment
    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    # Evaluate Model Performance
    print('Starting Evaluation...')
    start_time = time.time()
    mean_reward, success_rate, mean_steps = eval_model(env, eval_episodes=100, print_results=True)

    # Done Final Evaluation
    end_time = time.time()
    elapsed = end_time - start_time
    print(f'Evaluation Done In {elapsed:.2f} s ({elapsed/60:.2f} min)')
    print(f"Completion Rate: {success_rate:.2f}% | "
          f"Average Reward: {mean_reward:.4f} | "
          f"Average Steps: {mean_steps:.2f}"
    )

    # Close the Environment
    env.close()

