#============================================================================
# Import Modules
#============================================================================
# Imports for Environment
import gymnasium as gym
import minigrid
from minigrid.wrappers import *

# Imports for Generate Random Numbers / Epsilon Decay
import random
import math
# Import for Performance Tracking
import time

# Import for Various Mathematical, Vector and Matrix Functions
import numpy as np

from os.path import exists

# Import for Hyperparameter Tuning
import optuna

# Import for Experience Replay Memory
from collections import namedtuple, deque

# Imports for Neural Network
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# Imports for Writing toTensorboard
from torch.utils.tensorboard import SummaryWriter

# Check for GPU Availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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

#============================================================================
# SetUp the HyperParameters
#============================================================================
# ---- MODEL HYPERPARAMETERS ----
num_actions = 3                  # left, right, forward
steps_done = 0

# # ---- TRAINING HYPERPARAMETERS ----
# alpha = 0.0002                   # learning rate
# episodes = 3000                  # total episodes for training (per assignment spec)
# batch_size = 128                 # neural network batch size

# # ---- Q-LEARNING HYPERPARAMETERS ----
# gamma = 0.90                     # discounting rate

# # ---- EXPLORATION HYPERPARAMETERS for Epsilon Greedy Strategy ----
# start_epsilon = 1.0              # exploration probability at start
# stop_epsilon = 0.01              # minimum exploration probability
# decay_rate = 20000               # exponential decay rate for exploration probability

# ---- MEMORY HYPERPARAMETERS ----
target_update = 2000              # no. steps bet. updating target network
mem_size = 200000                  # no. experiences the memory can keep

# ---- TESTING HYPERPARAMETERS ----
# eval_episodes = 100              # no. episodes to be used for eval
filename = 'dqn_trained.pth'

#============================================================================
# Define and Create Convulutional Neural Network Model - RGB Images
#============================================================================
class CNN_DQN(nn.Module):

    def __init__(self, height, width, numActions, hiddenLayerSize=(512,)): 
        super(CNN_DQN, self).__init__()
     
        # h,w = 56, kernel_size = 5, stride = 2
        # input channels = 4 (one for each stacked frame)
        # output channels = 16
        
        # Input size is 4x56x56
        # Output size is 16x28x28
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Input is 16x26x26
        # Output is 32x14x14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Input is 32x11x11
        # Output is 32x4x4
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        # Flattened size is 32x4x4 = 512

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        linear_input_size = convw * convh * 32
        
        self.head = nn.Linear(linear_input_size, hiddenLayerSize[0])
        self.fc1 = nn.Linear(hiddenLayerSize[0], numActions)
        
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))) 

        x = F.relu(self.head(x.view(x.size(0), -1)))
        x = self.fc1(x)
        return x

    
# Define the Hidden Layer Size
hiddenLayerSize = (128,)

#============================================================================
# Epsilon-Greedy Exploration
#============================================================================
def select_action(state, explore_param, policy_net, writer, greedy=False):
    '''Select an action using epsilon-greedy exploration.
       If greedy=True, always act greedily (used for evaluation).'''
    global steps_done

    if greedy:
        with torch.no_grad():
            q_values = policy_net(state)
            action = q_values.max(1)[1].unsqueeze(0)
            return action
   
    # Extract Parameters
    start_epsilon = explore_param[0]
    stop_epsilon = explore_param[1]
    decay_rate = explore_param[2]

    # Generate a Random Number
    r = random.random()

    # Calculate the Epsilon Threshold
    epsilon_thres = stop_epsilon + (start_epsilon - stop_epsilon) * math.exp(-1. * steps_done / decay_rate)
    if writer is not None:
        writer.add_scalar('Epsilon', epsilon_thres, steps_done)

    # Compare Random Number to Epsilon Threshold
    if r > epsilon_thres:
        # Act Greedily Toward Q-Values of Policy Network given State
        with torch.no_grad():
            '''t.max(1) will return largest column value of each row.
            second column on max result is index of where max element was
            found, so we pick action with larger expected reward.'''
            return policy_net(state).max(1)[1].unsqueeze(0)
    else:
        # Select Random Action with Equal Probability
        return torch.tensor([[random.randrange(num_actions)]], device=device, dtype=torch.long)

#============================================================================
# Experience Replay Memory SetUp
#============================================================================
Transition = namedtuple('Transition', ('currentState', 'action', 'nextState', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        '''Save a Transition'''
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Initalise Experience Replay Memory
memory = ReplayMemory(mem_size)

#============================================================================
# Optimise Model
#============================================================================
def optimise_model(train_param, writer, optimiser, policy_net, target_net, memory):

    # Extract Parameters
    #alpha = train_param[0]
    gamma = train_param[1]
    batch_size = train_param[2]

    # Check if Replay memory has Stored Enough Experience
    if len(memory) < batch_size:
        return

    # ---- SAMPLE MINI-BATCH ----
    experience = memory.sample(batch_size)

    # Transpose the Batch (Convert Batch-Array of Experience to Experience of Batch-Arrays)
    batch = Transition(*zip(*experience))

    # ---- CALC ACTION-VALUE PREDICTED BY POLICY NETWORK ----
    state_batch = torch.cat(batch.currentState)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    '''Calculate the Action-Values for each State in Batch,
        and then gather the Q-Values for Action Associated with Specific State'''
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # ---- CALC TD-TARGET ESTIMATED BY TARGET NETWORK ----
    nf_next_states = torch.cat([s for s in batch.nextState if s is not None])

    # Initialise the next_state values to be zeroes
    next_state_values = torch.zeros(batch_size, device=device)

    # Compute a Mask of Non-Final States and Concatenate the Batch Elements
    nf_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.nextState)),
                            device=device, dtype=torch.bool)

    with torch.no_grad():
        # ---- DOUBLE DQN ----
        # Use POLICY network to select the best next action
        best_next_actions = policy_net(nf_next_states).max(1)[1].unsqueeze(1)

        # Use TARGET network only to evaluate that action's Q-value
        next_state_values[nf_mask] = target_net(nf_next_states).gather(1, best_next_actions).squeeze(1)

    # Compute the Expected Q-Values (TD-Target)
    TDtargets = (next_state_values * gamma) + reward_batch

    # ---- CALC LOSS using MINIMUM-SQUARED-ERROR CRITERION ----
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, TDtargets.unsqueeze(1))

    # TD-error: how far off the policy network's estimate was from the target
    TDerrors = TDtargets.unsqueeze(1) - state_action_values

    # ---- MAKE GRADIENT DESCENT STEP TO MINIMISE LOSS ----
    optimiser.zero_grad()
    loss.backward()

    # Clamp the Gradients in Policy Network to Range [-1, 1]
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)

    optimiser.step()

    # Log Loss to Tensorboard
    if writer is not None:
        writer.add_scalar('Loss', loss.item(), steps_done)
        writer.add_scalar('TDError', TDerrors.abs().mean().item(), steps_done)


#============================================================================
# Deep Q-Learning Function
#============================================================================
def deep_q_learning(env, train_param, explore_param, optimiser, policy_net, target_net, 
                    episodes=800, writer=None, memory=memory):

    # Declare Tracking Variables
    global steps_done
    episode_rewards = []
    episode_avg_steps = []
    success_rate = 0

    # Episode Loop
    for e in range(episodes):
        # Declare Epsiode Tracking Variables
        episode_steps = 0 
        episode_reward = 0.0

        # Reset the Environment
        current_obs, _ = env.reset()

        # Initialize the Frame Stack
        stacked_frames = deque([np.zeros((screen_height,screen_width), dtype=np.int32) 
                                            for i in range(stack_size)], maxlen=stack_size)

        # Preprocess the Observation to Obtain State
        current_state, stacked_frames = stack_frames(stacked_frames, current_obs, True)

        for s in range(0, max_steps):
            # Perform Epsilon-Greedy Action Selection
            action = select_action(current_state, explore_param, policy_net, writer, greedy=False)
            a = action.item()

            # Perform the Action in Environment
            next_obs, reward, done, truncated, _ = env.step(a)

            reward = torch.tensor([reward], device=device)

            # Increment Step Counters
            episode_steps += 1
            steps_done += 1
            # Calculate Accumulated Rewards
            episode_reward += reward

            # Preprocess the Observation to Obtain Next State
            if done or truncated:
                next_state = None
            else:
                next_state, stacked_frames = stack_frames(stacked_frames, next_obs, False)

            # Store Transition in Experience Replay Memory
            memory.push(current_state, action, next_state, reward )

            # Train Model
            if steps_done % 4 == 0:
                optimise_model(train_param, writer, optimiser, policy_net, target_net, memory)

            # Move to Next State
            current_state = next_state

            # Periodically Update Target Network
            if steps_done % target_update == 0:
                print(f'Updating target network at step {steps_done}')
                target_net.load_state_dict(policy_net.state_dict())

            # Increment the Successful Episodes
            if done: success_rate += 1

            # End Loop if Terminated or Truncated
            if done or truncated:
                break

        # Periodically Track Episode and Step Progress
        #if e % 100 == 0:
            #print(f'Episode {e}/{episodes} | Steps Done: {steps_done}')

        # Epsiode Rewards
        episode_rewards.append(episode_reward)
        episode_avg_steps.append(episode_steps) 



        # Write to Tensorboard upon Completion
        if writer is not None:
            writer.add_scalar('Reward/train', reward, steps_done)
            writer.add_scalar('Steps/train', s, steps_done)

    return np.mean(episode_rewards), success_rate/episodes*100, np.mean(episode_avg_steps) 

#============================================================================
# Evaluate Agent Performance
#============================================================================
def eval_model(env, eval_policy_net, eval_episodes=100, print_results=True):
    # Set Up Eval Counter Variables
    eval_success = 0.0
    total_steps = 0.0
    total_reward = 0.0

    # CHANGED: Remember the previous mode so evaluation does not leave a training
    # policy network in evaluation mode during Optuna.
    was_training = eval_policy_net.training

    # Set Policy to Evaluation Mode
    eval_policy_net.eval()

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
            for i in range(0, max_steps):
                # Select an Action
                action = select_action(current_state, _, eval_policy_net, writer= None, greedy=True)
                a = action.item()

                # Take Action
                next_obs, reward, done, truncated, info = env.step(a)

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

    # CHANGED: Restore the policy network to the mode it used before evaluation.
    eval_policy_net.train(was_training)

    return mean_reward, success_rate, mean_steps

#============================================================================
# Tune Hyperparameters using Optuna
#============================================================================
def objective(trial):
    # Tune Model Hyperparameters
    alpha = trial.suggest_float("alpha", 1e-5, 3e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.99)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    # CHANGED: Keep the original exponential epsilon decay, start with full
    # exploration, and let Optuna tune stop epsilon and the decay rate.
    start_epsilon = 1.0
    stop_epsilon = trial.suggest_float("stop_epsilon", 0.02, 0.1)
    decay_rate = trial.suggest_float("decay_rate", 50000, 100000)

    # Group Parameters
    train_param = [alpha, gamma, batch_size]
    explore_param = [start_epsilon, stop_epsilon, decay_rate]

    # CHANGED: Store every seed's evaluation results so the displayed values
    # and Optuna score represent all seeds rather than only the final seed.
    seed_rewards = []
    seed_success_rates = []
    seed_steps = []
    # Start Tuning Time
    start_time = time.time()

    global steps_done

    for seed in range(3): 
        # Reset Steps Done
        steps_done = 0 

        np.random.seed(seed)
        random.seed(seed)
        # CHANGED: Seed PyTorch as well so each trial uses the same three
        # reproducible network initialisations.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        memory = ReplayMemory(mem_size)

        # ---- RESET FRESH NETWORKS PER SEED ----
        # Instantiate the Policy Network and Target Network
        policy_net = CNN_DQN(screen_height, screen_width, num_actions, hiddenLayerSize).to(device)
        target_net = CNN_DQN(screen_height, screen_width, num_actions, hiddenLayerSize).to(device)
        # Copy Weights of Policy Network to Target Network
        target_net.load_state_dict(policy_net.state_dict())
        # Set Target Network to Eval Mode to not Update Parameters
        target_net.eval()
        # Adam Optimiser
        optimiser = optim.Adam(policy_net.parameters(), lr=alpha)

        # Make the Gym Environment
        env = gym.make('MiniGrid-Empty-8x8-v0')
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

        # CHANGED: Seed the environment for reproducible comparisons between trials.
        env.reset(seed=seed)

        # Search the Algorithm
        _,_,_ = deep_q_learning(env, train_param, explore_param, optimiser, policy_net, 
                                        target_net, episodes=500, writer=None, memory=memory) 
        # Evaluate Policy
        mean_reward, success_rate, mean_steps = eval_model(env, policy_net, eval_episodes=20, print_results=False) 
        
        # CHANGED: Save the results from this seed for aggregate reporting.
        seed_rewards.append(mean_reward)
        seed_success_rates.append(success_rate)
        seed_steps.append(mean_steps)

        # CHANGED: Weight reward by success rate so trials that work reliably
        # across all seeds rank above trials that only work for one or two seeds.
        partial_mean_reward = np.mean(seed_rewards)
        partial_mean_success = np.mean(seed_success_rates) / 100.0
        partial_score = partial_mean_reward * partial_mean_success
        trial.report(partial_score, seed)
        if trial.should_prune():
            raise optuna.TrialPruned()
        env.close()

    # CHANGED: Calculate aggregate trial results across all three seeds.
    overall_mean_reward = np.mean(seed_rewards)
    overall_success_rate = np.mean(seed_success_rates)
    overall_mean_steps = np.mean(seed_steps)
    trial_score = overall_mean_reward * (overall_success_rate / 100.0)

    elapsed = time.time() - start_time
    print(f'Trial Done In {elapsed:.2f} s ({elapsed/60:.2f} min)')
    # CHANGED: Print each seed and the aggregates instead of only the last seed.
    print(f'Seed Rewards: {seed_rewards}')
    print(f'Seed Success Rates: {seed_success_rates}')
    print(f'Seed Mean Steps: {seed_steps}')
    print(f'Overall Success Rate: {overall_success_rate:.2f}%')
    print(f'Overall Mean Steps: {overall_mean_steps:.2f}')
    print(f'Overall Mean Reward: {overall_mean_reward:.4f}')
    print(f'Optuna Trial Score: {trial_score:.4f}')

    # CHANGED: Return the success-weighted reward used to rank Optuna trials.
    return trial_score

#============================================================================
# Validate Best Hyperparameters Using Unseen Seeds
#============================================================================
def validate_hyperparameters(best_params, seeds=range(3, 8),
                             training_episodes=1000,
                             evaluation_episodes=20):

    global steps_done

    validation_rewards = []
    validation_success_rates = []
    validation_steps = []

    train_param = [
        best_params["alpha"],
        best_params["gamma"],
        best_params["batch_size"]
    ]

    # Use the unscaled decay rate because validation uses the same number
    # of training episodes as the Optuna trials.
    explore_param = [
        1.0,
        best_params["stop_epsilon"],
        best_params["decay_rate"]
    ]

    print("Validating best parameters on unseen seeds...")

    for seed in seeds:
        # Seed all random-number generators
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        steps_done = 0

        # Create Fresh Replay Memory
        validation_memory = ReplayMemory(mem_size)

        # Create Fresh Networks
        validation_policy_net = CNN_DQN(screen_height, screen_width, num_actions, hiddenLayerSize).to(device)
        validation_target_net = CNN_DQN(screen_height, screen_width, num_actions, hiddenLayerSize).to(device)

        validation_target_net.load_state_dict(validation_policy_net.state_dict())
        validation_target_net.eval()

        # Create a fresh optimiser
        validation_optimiser = optim.Adam(
            validation_policy_net.parameters(),
            lr=best_params["alpha"]
        )

        # Direction must remain in the observation
        validation_env = gym.make('MiniGrid-Empty-8x8-v0')
        validation_env = RGBImgPartialObsWrapper(validation_env)
        validation_env = ImgObsWrapper(validation_env)

        validation_env.reset(seed=seed)

        # Train from scratch
        deep_q_learning(
            validation_env,
            train_param,
            explore_param,
            validation_optimiser,
            validation_policy_net,
            validation_target_net,
            episodes=training_episodes,
            writer=None,
            memory=validation_memory
        )

        # Evaluate the trained policy
        mean_reward, success_rate, mean_steps = eval_model(
            validation_env,
            validation_policy_net,
            eval_episodes=evaluation_episodes,
            print_results=False
        )

        validation_rewards.append(mean_reward)
        validation_success_rates.append(success_rate)
        validation_steps.append(mean_steps)

        print(
            f"Validation seed {seed}: "
            f"reward={mean_reward:.4f}, "
            f"success={success_rate:.2f}%, "
            f"steps={mean_steps:.2f}"
        )

        validation_env.close()

    overall_reward = np.mean(validation_rewards)
    overall_success_rate = np.mean(validation_success_rates)
    overall_steps = np.mean(validation_steps)

    print(f"Validation Rewards: {validation_rewards}")
    print(f"Validation Success Rates: {validation_success_rates}")
    print(f"Validation Mean Steps: {validation_steps}")
    print(f"Overall Validation Reward: {overall_reward:.4f}")
    print(f"Overall Validation Success Rate: {overall_success_rate:.2f}%")
    print(f"Overall Validation Steps: {overall_steps:.2f}")

    return overall_reward, overall_success_rate, overall_steps


#============================================================================
# Run the Study
#============================================================================
if __name__ == "__main__":
        
    # Study the Model
    print("Start Hyperparameter Tuning...")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    # CHANGED: Increase the Optuna study to 100 trials.
    study.optimize(objective, n_trials= 50)
    
    # Display the Best Parameters
    print("Best Params:", study.best_params)
    print("Best Value:", study.best_value)

    # Extract the Best Parameters
    best = study.best_params

    # Log Best Parameters
    train_param = [best["alpha"], best["gamma"], best["batch_size"]]

    validation_reward, validation_success, validation_steps = (
        validate_hyperparameters(
            best_params=best,
            seeds=range(3, 8),
            training_episodes=1000,
            evaluation_episodes=20
        )
    )

    # final-training episodes, while keeping start epsilon fixed at 1.0.
    decay_rate = best["decay_rate"] * (3000 / 500)
    explore_param = [1.0, best["stop_epsilon"], decay_rate]

    # ---- RESET NETWORKS FOR FINAL TRAINING ----
    # Instantiate the Policy Network and Target Network
    policy_net = CNN_DQN(screen_height, screen_width, num_actions, hiddenLayerSize).to(device)
    target_net = CNN_DQN(screen_height, screen_width, num_actions, hiddenLayerSize).to(device)
      # Copy Weights of Policy Network to Target Network
    target_net.load_state_dict(policy_net.state_dict())
    # Set Target Network to Eval Mode to not Update Parameters
    target_net.eval()
    # Adam Optimiser
    optimiser = optim.Adam(policy_net.parameters(), lr=best["alpha"])

    # Gym Environment
    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
        
    # SetUp Tensorboard
    writer = SummaryWriter()

    memory = ReplayMemory(mem_size)

    # Start Training
    print('Start training...')
    start_time = time.time()

    steps_done = 0

    # Train the Model
    mean_reward, success_rate, mean_steps = deep_q_learning(env, train_param, explore_param, optimiser, policy_net, 
                                     target_net, episodes=3000, writer=writer, memory=memory)

    # Done Training
    print('Done Training...')
    end_time = time.time()
    elapsed = end_time - start_time
    print(f'Training Done In {elapsed:.2f} s ({elapsed/60:.2f} min)')
    print(f'Mean Steps Taken: {mean_steps}')
    print(f'Mean Reward: {mean_reward}')
    print(f'Success Rate: {success_rate}')

    
    # Flush Remaining Data
    writer.flush()
    writer.close()

    # Save the Trained Model
    torch.save(policy_net.state_dict(), filename)
    
    # Reload the Newly Trained Model
    if exists(filename):
        final_policy_net = CNN_DQN(screen_height, screen_width, num_actions, hiddenLayerSize).to(device)
        final_policy_net.load_state_dict(torch.load(filename, map_location=device))
        print(f'Loaded trained model from {filename}')
    else:
        raise FileNotFoundError(f'No saved model found at {filename}.')

    # Evaluate Model Performance
    print('Starting Evaluation...')
    start_time = time.time()
    final_policy_net.eval()
    mean_reward, success_rate, mean_steps = eval_model(env, final_policy_net, eval_episodes=100, print_results=True)

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



