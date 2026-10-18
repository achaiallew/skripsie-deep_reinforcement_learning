#============================================================================
# Imports
#============================================================================
# Imports for Pytorh
import torch
from torch import nn
import torch.nn.functional as F

# Imports for Environment
import gymnasium as gym
from minigrid.wrappers import *

# Imports for Post Training Quantization
import copy
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, Float8DynamicActivationInt4WeightConfig 
from torchao.quantization import Float8StaticActivationFloat8WeightConfig, Float8WeightOnlyConfig, Int4WeightOnlyConfig  
from torchao.quantization import Int8DynamicActivationInt8WeightConfig,  Int8DynamicActivationIntxWeightConfig 
from torchao.quantization import Int8StaticActivationInt8WeightConfig, Int8WeightOnlyConfig, quantize_

import image_dqn

import os
from os.path import exists

import time

import numpy as np

# =====================================================
# Settings
# =====================================================

model_trained = "dqn_trained.pth"     
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

# =====================================================
# Post Training Quantization
# =====================================================
def quantize_models(pnet, config, config_name):
    # Create a Copy to Quantize
    quant_pnet = copy.deepcopy(pnet)

    # Quantize the Model
    quantize_(
        quant_pnet,
        config
    )

    # Evaluate 
    quant_reward, quant_rate, quant_steps = image_dqn.eval_model(env, quant_pnet, print_results=False)

    print("Post Training Quantization Policy Network Evaluation:"
        f"\nCompletion Rate: {quant_rate:.2f}% | "
        f"\nAverage Reward: {quant_reward:.4f} | "
        f"\nAverage Steps: {quant_steps:.2f}")

    # Save Model
    torch.save(quant_pnet.state_dict(), f"{config_name}_PTQ.pth")

    # Compare File Sizes
    size = os.path.getsize("dqn_trained.pth") / 1024
    quant_size = os.path.getsize(f"{config_name}_PTQ.pth") / 1024

    print(f"Size Reduction: {size/quant_size:.2f}x ({size:.2f}KB -> {quant_size:.2f}KB)")

    # Compare Model Inference
    input_tensor = torch.randn(1, 4, screen_height, screen_width)

    baseline_results = benchmark_inference(
        pnet,
        input_tensor
    )

    ptq_results = benchmark_inference(
        quant_pnet,
        input_tensor
    )

    for name, results in [
        ("Baseline", baseline_results),
        (f"PTQ for {config_name}", ptq_results)
    ]:

        print(f"\n{name}")
        print(f"Mean:       {results['mean_ms']:.4f} ms")
        print(f"Std:        {results['std_ms']:.4f} ms")
        print(f"Median:     {results['median_ms']:.4f} ms")
        print(f"Min:        {results['min_ms']:.4f} ms")
        print(f"Max:        {results['max_ms']:.4f} ms")
        print(f"Throughput: {results['throughput']:.2f} inferences/s")

    # Speedup
    baseline_time = baseline_results["mean_ms"]
    ptq_time = ptq_results["mean_ms"]

    print(f"\nBaseline → PTQ {config_name} Speedup: {baseline_time / ptq_time:.2f}x")

# =====================================================
# Compare Inference
# =====================================================
def benchmark_inference(model, input_tensor, num_warmup=100, num_runs=1000):

    model.eval()

    with torch.inference_mode():

        # Warm-up
        for _ in range(num_warmup):
            model(input_tensor)

        times = []

        for _ in range(num_runs):
            start = time.perf_counter()

            model(input_tensor)

            end = time.perf_counter()

            times.append((end - start) * 1000)

    times = np.array(times)

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "median_ms": np.median(times),
        "throughput": 1000 / np.mean(times)
    }

# =====================================================
# Run the Code for All Models
# =====================================================
if __name__ == "__main__":
    models = {"F8WO": Float8WeightOnlyConfig(),
              "I8DAI8W": Int8DynamicActivationInt8WeightConfig(),
              "I8DAI4W": Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4),
              "I8WO": Int8WeightOnlyConfig()
              }
    
    for name, config in models.items():
        try:
            quantize_models(policy_net, config, name)
        except (AssertionError, RuntimeError) as e:
            print(f"\nSkipping {name}: {e}")