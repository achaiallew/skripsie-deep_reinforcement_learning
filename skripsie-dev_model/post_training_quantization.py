import torch
from torch import nn
import torch.nn.functional as F

# Imports for Environment
import gymnasium as gym
from minigrid.wrappers import *

import copy
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, Int8WeightOnlyConfig, quantize_

import deep_q_network

import os
from os.path import exists

import time

# =====================================================
# Settings
# =====================================================

model_trained = "dqn_trained.pth"      # <-- change if necessary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- MODEL HYPERPARAMETERS ----
num_actions = 3                  # left, right, forward
input_size = 53                 # size of flattened input state (7x7 matrix of tile IDs)

# Make the Gym Environment
env = gym.make('MiniGrid-Empty-8x8-v0', render_mode=None).unwrapped

# =====================================================
# Network
# =====================================================
class DQN(nn.Module):

    def __init__(self, inputSize, numActions, hiddenLayerSize=(128,128)):
        super().__init__()

        self.fc1 = nn.Linear(inputSize, hiddenLayerSize[0])
        self.fc2 = nn.Linear(hiddenLayerSize[0], hiddenLayerSize[1])
        self.fc3 = nn.Linear(hiddenLayerSize[1], numActions)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
# Update Hidden Layer Size
hiddenLayerSize = (128, 128)

# =====================================================
# Load Model
# =====================================================
 # Reload the Newly Trained Model
if exists(model_trained):
    policy_net = DQN(input_size, num_actions, hiddenLayerSize).to(device)
    policy_net.load_state_dict(torch.load(model_trained, map_location=device))
    policy_net.cpu()
    policy_net.eval()
    print(f'Loaded trained model from {model_trained}')
else:
    raise FileNotFoundError(f'No saved model found at {model_trained}.')

# Create a Copy to Quantize
quantized_policy_net = copy.deepcopy(policy_net)
wo_quantized_policy_net = copy.deepcopy(policy_net)

# Quantize the Model - Weight Only
quantize_(
    wo_quantized_policy_net,
    Int8WeightOnlyConfig()
)
# Quantize the Model - Dynamic Activations and Weights
quantize_(
    quantized_policy_net,
    Int8DynamicActivationInt8WeightConfig()
)

# Evaluate All
float_reward, float_rate, float_steps = deep_q_network.eval_model(env, policy_net)
quant_reward, quant_rate, quant_steps = deep_q_network.eval_model(env, quantized_policy_net)
wo_quant_reward, wo_quant_rate, wo_quant_steps = deep_q_network.eval_model(env, wo_quantized_policy_net)

# Print Results
print("Policy Network Evaluation:"
        f"\nCompletion Rate: {float_rate:.2f}% | "
        f"\nAverage Reward: {float_reward:.4f} | "
        f"\nAverage Steps: {float_steps:.2f}"
    )

print("Post Training Quantization Policy Network Evaluation:"
        f"\nCompletion Rate: {quant_rate:.2f}% | "
        f"\nAverage Reward: {quant_reward:.4f} | "
        f"\nAverage Steps: {quant_steps:.2f}"
    )

print("Post Training Quantization (Weight Only) Policy Network Evaluation:"
        f"\nCompletion Rate: {wo_quant_rate:.2f}% | "
        f"\nAverage Reward: {wo_quant_reward:.4f} | "
        f"\nAverage Steps: {wo_quant_steps:.2f}"
    )

# Save the Quantized Model
torch.save(quantized_policy_net.state_dict(), "dqn_quantized.pth")
torch.save(wo_quantized_policy_net.state_dict(), "dqn_quantized_wo.pth")


# Compare File Sizes
original_size = os.path.getsize("dqn_trained.pth") / 1024**2
quantized_size = os.path.getsize("dqn_quantized.pth") / 1024**2
wo_quantized_size = os.path.getsize("dqn_quantized_wo.pth") / 1024**2

print(
    f"Size Reduction: {original_size / quantized_size:.2f}x ({original_size:.2f}MB -> {quantized_size:.2f}MB)"
)
print(
    f"Size Reduction (Weight Only): {original_size / wo_quantized_size:.2f}x ({original_size:.2f}MB -> {wo_quantized_size:.2f}MB)"
)

# Compare Model Speed Up
original_model = policy_net
quantized_model = quantized_policy_net
wo_quantized_model = wo_quantized_policy_net

# Get Example Inputs
example_inputs = torch.randn(1, input_size)
wo_example_inputs = torch.randn(1, input_size)


# Throughput: original model
start = time.perf_counter()

for _ in range(1000):
    with torch.inference_mode():
        _ = original_model(example_inputs)

original_time = time.perf_counter() - start

# Throughput: Quantized (W8A8-INT) model
start = time.perf_counter()

for _ in range(1000):
    with torch.inference_mode():
        _ = quantized_model(example_inputs)

quantized_time = time.perf_counter() - start

print(f"Speedup: {original_time / quantized_time:.2f}x")


# Throughput: original model
start = time.perf_counter()

for _ in range(1000):
    with torch.inference_mode():
        _ = original_model(wo_example_inputs)

original_time = time.perf_counter() - start

# Throughput: Quantized (W8A8-INT) model
start = time.perf_counter()

for _ in range(1000):
    with torch.inference_mode():
        _ = wo_quantized_model(wo_example_inputs)

wo_quantized_time = time.perf_counter() - start

print(f"Speedup (Weight Only): {original_time / wo_quantized_time:.2f}x")