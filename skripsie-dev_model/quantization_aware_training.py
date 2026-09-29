#============================================================================
# Imports
#============================================================================
import copy
import os
from os.path import exists
import time

# Imports for Environment
import gymnasium as gym
from minigrid.wrappers import *

# Imports for Pytorh
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

# Imports for Quantization Aware Training
from torchao.quantization import quantize_, Int8DynamicActivationIntxWeightConfig
from torchao.quantization.granularity import PerAxis
from torchao.quantization.qat import QATConfig

# Imports for Writing to Tensorboard
from torch.utils.tensorboard import SummaryWriter

#Imports from DQN 
from deep_q_network import (ReplayMemory, deep_q_learning, eval_model, device)

#============================================================================
# Settings
#============================================================================
# Load Model
model_trained = "dqn_trained.pth" 

# ---- MODEL HYPERPARAMETERS ----
num_actions = 3                  
input_size = 53
mem_size = 200000

# Hidden Layer Size
hiddenLayerSize = (128, 128)

#============================================================================
# Network
#============================================================================
class DQN_Quant(nn.Module):

    def __init__(self, inputSize, numActions, hiddenLayerSize=(128,128)):
        super().__init__()

        self.fc1 = nn.Linear(inputSize, hiddenLayerSize[0])
        self.fc2 = nn.Linear(hiddenLayerSize[0], hiddenLayerSize[1])
        self.fc3 = nn.Linear(hiddenLayerSize[1], numActions)

    def forward(self, x):
        x = x.to(self.fc1.weight.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


#============================================================================
# Load Original Model
#============================================================================
if exists(model_trained):
    policy_net = DQN_Quant(input_size, num_actions, hiddenLayerSize).to(device)
    policy_net.load_state_dict(torch.load(model_trained, map_location=device))
    policy_net.eval()
    print(f'Loaded trained model from {model_trained}')
else:
    raise FileNotFoundError(f'No saved model found at {model_trained}.')

#============================================================================
# Create QAT Policy and Target Networks
#============================================================================
# Create a Copy to Quantize
qat_policy_net = copy.deepcopy(policy_net)

# Create a Copy for the Target Network 
qat_target_net = copy.deepcopy(policy_net)

#============================================================================
# Configure Fake Quantization
#============================================================================
# Configure INT8 Activations and INT4 Weights
base_config = Int8DynamicActivationIntxWeightConfig(
    weight_dtype=torch.int4,
    weight_granularity=PerAxis(0)
)

# Prepare Fake Quantization
qat_prepare_config = QATConfig(
    base_config,
    step="prepare",
)

# Convert Fake Quantization (after training)
qat_convert_config = QATConfig(
    base_config,
    step="convert",
)

#============================================================================
# Prepare Policy and Target Networks for QAT
#============================================================================
# Fake Quatization of the Policy Network
quantize_(qat_policy_net, qat_prepare_config)

# Fake Quatization of the Target Network
quantize_(qat_target_net, qat_prepare_config)

# Set Target Network to Eval Mode to not Update Parameters
qat_target_net.eval()

#============================================================================
# Setup HyperParameters from Previously Trained Model
#============================================================================
# Best Learning Rate
# alpha = 0.00013655199655359068

# ---- TRAINING HYPERPARAMETERS ----
alpha = 0.000013655199655359068
gamma = 0.8435494470444489
batch_size = 128
train_param = [alpha, gamma, batch_size]

# ---- EXPLORATION HYPERPARAMETERS ----
start_epsilon = 0.1
stop_epsilon = 0.01
decay_rate = 10000
explore_param = [start_epsilon, stop_epsilon, decay_rate]

#============================================================================
# Configure Fake Quantization Training Parameters
#============================================================================
# Memory
qat_memory = ReplayMemory(mem_size)

# Episode Size
qat_episodes = 500

# Adam Optimiser
qat_optimiser = optim.Adam(qat_policy_net.parameters(), alpha)

# Make the Gym Environment
env = gym.make('MiniGrid-Empty-8x8-v0', render_mode=None).unwrapped

# SetUp Tensorboard
writer = SummaryWriter()

#============================================================================
# Quantization Aware Training - Fine Tuning
#============================================================================
# Start QAT
print('Start Quantization Aware Training (Fine-Tuning)...')
start_time = time.time()

# Train with QAT
qat_policy_net.train()
mean_reward, success_rate, mean_steps = deep_q_learning(env, train_param, explore_param, qat_optimiser, qat_policy_net, 
                                                        qat_target_net, episodes=qat_episodes, writer=writer, memory=qat_memory)

# Done Training
print('Done Quantization Aware Training (Fine-Tuning)...')
end_time = time.time()

# Determine Training Time
elapsed = end_time - start_time

# Print Results to Screen
print(f'Training Done In {elapsed:.2f} s ({elapsed/60:.2f} min)')
print(f'Mean Steps Taken: {mean_steps}')
print(f'Mean Reward: {mean_reward}')
print(f'Success Rate: {success_rate}')

# Flush Remaining Data
writer.flush()
writer.close()

#============================================================================
# Convert QAT Model
#============================================================================
# Copy Model to Convert
qat_quantized_net = copy.deepcopy(qat_policy_net)
qat_quantized_net.eval()
# Move to CPU for inference quantization
qat_quantized_net.cpu()

# Remove the Fake-Quantization Wrappers (Convert)
quantize_(qat_quantized_net, qat_convert_config)

# Save the Trained Model
torch.save(qat_quantized_net.state_dict(), "dqn_qat_quantized.pth")

#============================================================================
# Evaluate Final QAT Model
#============================================================================
# Evaluate Converted Model Performance
print('Starting Evaluation...')
start_time = time.time()

# EVALUATION
qat_reward, qat_success, qat_steps = eval_model(env, qat_quantized_net, eval_episodes=100, print_results=True)


# Done Final Evaluation
end_time = time.time()
elapsed = end_time - start_time

# Print Results to Screen
print(f'Evaluation Done In {elapsed:.2f} s ({elapsed/60:.2f} min)')
print(f'Evaluation Done In {elapsed:.2f} s ({elapsed/60:.2f} min)')
print(f"Completion Rate: {qat_success:.2f}% | "
        f"Average Reward: {qat_reward:.4f} | "
        f"Average Steps: {qat_steps:.2f}"
)

#============================================================================
# Compare Model File Sizes
#============================================================================
# Compare File Sizes
original_size = os.path.getsize("dqn_trained.pth") / 1024**2
qat_size = os.path.getsize("dqn_qat_quantized.pth") / 1024**2

print(
    f"Size Reduction (Quantized): {original_size / qat_size:.2f}x ({original_size:.2f}MB -> {qat_size:.2f}MB)"
)

#============================================================================
# Compare Pure Model Inference Speed
#============================================================================# Both models must be on CPU for a fair comparison
fp32_model = copy.deepcopy(policy_net).cpu()
fp32_model.eval()

example_input = torch.randn(
    1,
    input_size
)

# Warm-up
with torch.inference_mode():
    for _ in range(100):
        fp32_model(example_input)
        qat_quantized_net(example_input)

# Time FP32 model
start_time = time.perf_counter()

with torch.inference_mode():
    for _ in range(10000):
        fp32_model(example_input)

fp32_time = time.perf_counter() - start_time

# Time quantized QAT model
start_time = time.perf_counter()

with torch.inference_mode():
    for _ in range(10000):
        qat_quantized_net(example_input)

quantized_time = time.perf_counter() - start_time

print("\nInference Speed Comparison:")

print(f"FP32 time: {fp32_time:.4f} s")

print(
    f"INT4 QAT time: "
    f"{quantized_time:.4f} s"
)

print(
    f"Speed-up: "
    f"{fp32_time / quantized_time:.2f}x"
)


#============================================================================
# Close Environment
#============================================================================
env.close()