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

# Import Numpy for Model Inference
import numpy as np


# Imports for Quantization Aware Training
from torchao.quantization import Int8DynamicActivationIntxWeightConfig
from torchao.quantization import IntxWeightOnlyConfig, quantize_
from torchao.quantization.granularity import PerAxis
from torchao.quantization.qat import QATConfig

# Imports for Writing to Tensorboard
from torch.utils.tensorboard import SummaryWriter

#Imports from DQN 
from image_dqn import (ReplayMemory, deep_q_learning, eval_model, device)

#============================================================================
# Settings
#============================================================================
# Load Model
model_trained = "dqn_trained.pth" 

# ---- MODEL HYPERPARAMETERS ----
num_actions = 3                  
screen_height = 56
screen_width = 56
mem_size = 200000

# Define the Hidden Layer Size
hiddenLayerSize = (128,)

#============================================================================
# Create the Network
#============================================================================
class CNN_DQN_Quant(nn.Module):

    def __init__(self, height, width, numActions, hiddenLayerSize=(512,)): 
        super(CNN_DQN_Quant, self).__init__()
     
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

#============================================================================
# Load Original Model
#============================================================================
if exists(model_trained):
    policy_net = CNN_DQN_Quant(screen_height, screen_width, num_actions, hiddenLayerSize).to(device)
    policy_net.load_state_dict(torch.load(model_trained, map_location=device))
    policy_net.eval()
    print(f'Loaded trained model from {model_trained}')
else:
    raise FileNotFoundError(f'No saved model found at {model_trained}.')

#============================================================================
# Quantization Aware Training
#============================================================================
def quantize_models(pnet, config, config_name):

    # Create a Copy for the Policy Network
    qat_policy_net = copy.deepcopy(pnet)

    # Create a Copy for the Target Network
    qat_target_net = copy.deepcopy(pnet)

    #============================================================================
    # Configure Fake Quantization
    #============================================================================
    qat_prepare_config = QATConfig(
        config,
        step="prepare",
    )

    qat_convert_config = QATConfig(
        config,
        step="convert",
    )

    #============================================================================
    # Prepare Policy and Target Networks for QAT
    #============================================================================
    # Fake Quantization of the Policy Network
    quantize_(qat_policy_net, qat_prepare_config)

    # Fake Quantization of the Target Network
    quantize_(qat_target_net, qat_prepare_config)

    # Set Target Network to Eval Mode to not Update Parameters
    qat_target_net.eval()

    #============================================================================
    # Setup HyperParameters from Previously Trained Model
    #============================================================================
    # Best Learning Rate
    # alpha = 0.00012488734110133683

    # ---- TRAINING HYPERPARAMETERS ----
    alpha = 0.000012488734110133683
    gamma = 0.9191933975967473
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

    #============================================================================
    # Quantization Aware Training - Fine Tuning
    #============================================================================
    print(f'\nStart Quantization Aware Training for {config_name} (Fine-Tuning)...')
    start_time = time.time()

    # SetUp Tensorboard
    writer = SummaryWriter(log_dir=f"runs/QAT_{config_name}")

    # Train with QAT
    qat_policy_net.train()
    mean_reward, success_rate, mean_steps = deep_q_learning(
        env, train_param, explore_param, qat_optimiser, qat_policy_net,
        qat_target_net, episodes=qat_episodes, writer=writer, memory=qat_memory
    )

    # Done Training
    print(f'Done Quantization Aware Training for {config_name} (Fine-Tuning)...')
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
    output_file = f"{config_name}_QAT.pth"
    torch.save(qat_quantized_net.state_dict(), output_file)

    #============================================================================
    # Evaluate Final QAT Model
    #============================================================================
    print(f'Starting Evaluation for {config_name}...')
    start_time = time.time()

    # EVALUATION
    qat_reward, qat_success, qat_steps = eval_model(
        env, qat_quantized_net, eval_episodes=100, print_results=False
    )

    # Done Final Evaluation
    end_time = time.time()
    elapsed = end_time - start_time

    # Print Results to Screen
    print(f'Evaluation Done In {elapsed:.2f} s ({elapsed/60:.2f} min)')
    print(f"Completion Rate: {qat_success:.2f}% | "
          f"Average Reward: {qat_reward:.4f} | "
          f"Average Steps: {qat_steps:.2f}")

    #============================================================================
    # Compare Model File Sizes
    #============================================================================
    # Compare File Sizes
    original_size = os.path.getsize("dqn_trained.pth") / 1024
    qat_size = os.path.getsize(output_file) / 1024

    print(
        f"Size Reduction: {original_size / qat_size:.2f}x "
        f"({original_size:.2f}KB -> {qat_size:.2f}KB)"
    )

    #============================================================================
    # Compare Model Inference
    #============================================================================
    input_tensor = torch.randn(1, 4, screen_height, screen_width)

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

    # Both models must be on CPU for a fair comparison
    fp32_model = copy.deepcopy(pnet).cpu()
    fp32_model.eval()

    qat_quantized_net.cpu()
    qat_quantized_net.eval()

    baseline_results = benchmark_inference(
        fp32_model,
        input_tensor
    )

    qat_results = benchmark_inference(
        qat_quantized_net,
        input_tensor
    )

    for name, results in [
        ("Baseline", baseline_results),
        (f"QAT for {config_name}", qat_results)
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
    qat_time = qat_results["mean_ms"]

    print(
        f"\nBaseline → QAT {config_name} Speedup: "
        f"{baseline_time / qat_time:.2f}x"
    )

#============================================================================
# Run the Code for All Models
#============================================================================
if __name__ == "__main__":

    # Make the Gym Environment
    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    models = {
        "I8DAI8W": Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int8,
            weight_granularity=PerAxis(0),
        ),
        "I8DAI4W": Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerAxis(0),
        ),
        "I8WO": IntxWeightOnlyConfig(
            weight_dtype=torch.int8,
            granularity=PerAxis(0),   # note: `granularity`, not `weight_granularity`
        ),
    }

    for name, config in models.items():
        try:
            quantize_models(policy_net, config, name)
        except (AssertionError, RuntimeError) as e:
            print(f"\nSkipping {name}: {e}")

    #============================================================================
    # Close Environment
    #============================================================================
    env.close()