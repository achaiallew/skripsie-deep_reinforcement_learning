import torch
from torch import nn
import torch.nn.functional as F

# Imports for Environment
import gymnasium as gym
from minigrid.wrappers import *

import copy
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, Int8WeightOnlyConfig, quantize_

import image_dqn

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
screen_height = 56
screen_width = 56

# Gym Environment
env = gym.make('MiniGrid-Empty-8x8-v0')
env = RGBImgPartialObsWrapper(env)
env = ImgObsWrapper(env)

# =====================================================
# Network
# =====================================================
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

# =====================================================
# Load Model
# =====================================================
 # Reload the Newly Trained Model
if exists(model_trained):
    policy_net = CNN_DQN(screen_height, screen_width, num_actions, hiddenLayerSize).to(device)
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
float_reward, float_rate, float_steps = image_dqn.eval_model(env, policy_net)
quant_reward, quant_rate, quant_steps = image_dqn.eval_model(env, quantized_policy_net)
wo_quant_reward, wo_quant_rate, wo_quant_steps = image_dqn.eval_model(env, wo_quantized_policy_net)

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
example_inputs = torch.randn(1, 4, screen_height, screen_width)
wo_example_inputs = torch.randn(1, 4, screen_height, screen_width)


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


# Throughput: Weight Only model
start = time.perf_counter()

for _ in range(1000):
    with torch.inference_mode():
        _ = original_model(wo_example_inputs)

original_time = time.perf_counter() - start


# Throughput: Weight Only Quantized model
start = time.perf_counter()

for _ in range(1000):
    with torch.inference_mode():
        _ = wo_quantized_model(wo_example_inputs)

wo_quantized_time = time.perf_counter() - start

print(f"Speedup (Weight Only): {original_time / wo_quantized_time:.2f}x")