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

# Imports for Writing to Tensorboard
from torch.utils.tensorboard import SummaryWriter

#Imports from DQN 
from image_dqn import (ReplayMemory, deep_q_learning, eval_model, device)
import image_dqn

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

# ---- QUANTIZATION HYPERPARAMETERS ----
# Momentum of the Moving Average for the Activation Ranges (Section 2.5)
act_momentum = 0.01

# Total Number of Fine-Tuning Episodes
qat_episodes = 500

# Episodes Trained in Float Before Quantization Starts (Quantization Delay)
quant_delay_episodes = 50

# Episode at Which the Batch Norm Statistics are Frozen (Section 3.2.2)
bn_freeze_episode = 350

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
# Simulated Quantizer (Section 2.1 and 2.4)
#============================================================================
# Quantize -> Clamp -> Dequantize (Eq. 1-3) with a Straight Through
# Estimator for the Backward Pass (Eq. 14-15)
def SimQuant(delta, z, x, bits=8):
    n_levels = 2 ** bits

    # Representable Float Range (Eq. 14)
    x_min = (0 - z) * delta
    x_max = (n_levels - 1 - z) * delta

    # Forward: The Real Quantizer (Eq. 1-3)
    x_int = torch.round(x / delta) + z
    x_q = torch.clamp(x_int, 0, n_levels - 1)
    x_deq = (x_q - z) * delta

    # Backward: Pretend the Function was clamp(x, x_min, x_max) (Eq. 14-15)
    x_c = torch.clamp(x, x_min, x_max)

    # Value Equals x_deq, Gradient Only Flows Through x_c
    return x_c + (x_deq - x_c).detach()

#============================================================================
# Quantizer Parameters (Section 2.1 and 2.5)
#============================================================================
# Convert a Min/Max Range to a Scale (delta) and Zero-Point (z)
def compute_qparams(x_min, x_max, bits=8, eps=1e-8):
    n_levels = 2 ** bits

    # Relax the Range to Include Zero so Zero is Quantized Exactly
    x_min = torch.clamp(x_min, max=0.0)
    x_max = torch.clamp(x_max, min=0.0)

    # Scale and Zero-Point (Guard Against a Constant Tensor)
    delta = (x_max - x_min).clamp(min=eps) / (n_levels - 1)
    z = torch.round(-x_min / delta).clamp(0, n_levels - 1)

    return delta, z

#============================================================================
# Weight Quantizer (Per-Channel, Actual Min/Max Every Forward Pass)
#============================================================================
class WeightQuant(nn.Module):

    def __init__(self, bits=8):
        super(WeightQuant, self).__init__()
        self.bits = bits
        self.enabled = True

    # One Scale and Zero-Point for Every Output Channel
    def get_qparams(self, w):
        with torch.no_grad():
            flat = w.reshape(w.shape[0], -1)
            delta, z = compute_qparams(flat.min(dim=1).values, flat.max(dim=1).values, self.bits)

            # Make Broadcastable Against the Weight
            shape = [-1] + [1] * (w.dim() - 1)
            return delta.view(shape), z.view(shape)

    def forward(self, w):
        if not self.enabled:
            return w

        delta, z = self.get_qparams(w)
        return SimQuant(delta, z, w, self.bits)

#============================================================================
# Activation Quantizer (Per-Layer, Moving Average of Min/Max)
#============================================================================
class ActQuant(nn.Module):

    def __init__(self, bits=8, momentum=0.01):
        super(ActQuant, self).__init__()
        self.bits = bits
        self.momentum = momentum
        self.enabled = True

        # Buffers are Saved in the state_dict (so the Target Network Gets
        # Them) but are Ignored by the Optimiser
        self.register_buffer("min_val", torch.zeros(()))
        self.register_buffer("max_val", torch.zeros(()))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    def update_range(self, x):
        with torch.no_grad():
            batch_min, batch_max = x.min(), x.max()

            # Seed the Range From the First Batch
            if not self.initialized:
                self.min_val.copy_(batch_min)
                self.max_val.copy_(batch_max)
                self.initialized.fill_(True)
            else:
                m = self.momentum
                self.min_val.mul_(1 - m).add_(m * batch_min)
                self.max_val.mul_(1 - m).add_(m * batch_max)

    def get_qparams(self):
        with torch.no_grad():
            return compute_qparams(self.min_val, self.max_val, self.bits)

    def forward(self, x):
        # Only Update the Range in Training Mode. A Batch of One is an Action
        # Selection Step (Epsilon-Greedy), not a Training Batch, so Skip It
        if self.training and x.size(0) > 1:
            self.update_range(x)

        # Pass Through Until Quantization is On and the Range is Known
        if not self.enabled or not self.initialized:
            return x

        delta, z = self.get_qparams()
        return SimQuant(delta, z, x, self.bits)

#============================================================================
# Quantized Conv + Batch Norm Layer (Section 3.2.2, Figure 9)
#============================================================================
class QuantConvBN(nn.Module):

    def __init__(self, conv, bn, bits=8):
        super(QuantConvBN, self).__init__()

        # Keep the Original Layers so the Weights are Reused Directly
        self.conv = conv
        self.bn = bn
        self.weight_quant = WeightQuant(bits)

        # Batch Norm Cancels the Conv Bias, so Absorb it Into the Running Mean
        # and Remove it. Otherwise it Never Gets a Gradient (grad is None)
        if self.conv.bias is not None:
            with torch.no_grad():
                self.bn.running_mean.sub_(self.conv.bias)
            self.conv.bias = None

        # Once True, Batch Norm Uses the Long-Term Statistics During Training
        self.bn_frozen = False

    # Batch Norm Folded Into the Weights and Bias (Eq. 20 and 21)
    def folded_params(self):
        with torch.no_grad():
            sigma = torch.sqrt(self.bn.running_var + self.bn.eps)
            w_fold = self.conv.weight * (self.bn.weight / sigma).view(-1, 1, 1, 1)
            bias_fold = self.bn.bias - self.bn.weight * self.bn.running_mean / sigma
        return w_fold, bias_fold

    def forward(self, x):
        conv = self.conv
        bn = self.bn

        # Batch Statistics are Only Used While Training and Before Freezing
        use_batch_stats = self.training and not self.bn_frozen and x.size(0) > 1

        # Long-Term Standard Deviation and the Folded Weights (Eq. 20)
        sigma = torch.sqrt(bn.running_var + bn.eps)
        w_fold = conv.weight * (bn.weight / sigma).view(-1, 1, 1, 1)

        # Quantize the Folded Weights, then Convolve
        w_q = self.weight_quant(w_fold)
        y = F.conv2d(x, w_q, None, conv.stride, conv.padding)

        if use_batch_stats:
            # Batch Statistics From the Unfolded Convolution
            with torch.no_grad():
                y_raw = F.conv2d(x, conv.weight, conv.bias, conv.stride, conv.padding)
                mu_b = y_raw.mean(dim=(0, 2, 3))
                var_b = y_raw.var(dim=(0, 2, 3), unbiased=False)
                sigma_b = torch.sqrt(var_b + bn.eps)

                # Update the Long-Term Statistics (Moving Averages)
                n = y_raw.numel() / y_raw.size(1)
                m = bn.momentum
                bn.running_mean.mul_(1 - m).add_(m * mu_b)
                bn.running_var.mul_(1 - m).add_(m * var_b * n / (n - 1))

            # Undo the Correction so the Output Matches Regular Batch Norm (Eq. 22-27)
            c = sigma_b / sigma
            y = y / c.view(1, -1, 1, 1)
            bias = bn.bias - bn.weight * mu_b / sigma_b
        else:
            # Frozen / Inference: Long-Term Statistics Only (Eq. 28-31)
            bias = bn.bias - bn.weight * bn.running_mean / sigma

        return y + bias.view(1, -1, 1, 1)

#============================================================================
# Quantized Linear Layer
#============================================================================
class QuantLinear(nn.Module):

    def __init__(self, linear, bits=8):
        super(QuantLinear, self).__init__()
        self.linear = linear
        self.weight_quant = WeightQuant(bits)

    def folded_params(self):
        with torch.no_grad():
            return self.linear.weight, self.linear.bias

    def forward(self, x):
        w_q = self.weight_quant(self.linear.weight)
        return F.linear(x, w_q, self.linear.bias)

#============================================================================
# Quantization Aware Network
#============================================================================
# act_bits = None Quantizes the Weights Only (Activations Stay in Float)
class CNN_DQN_QAT(nn.Module):

    def __init__(self, float_net, weight_bits=8, act_bits=8, momentum=0.01):
        super(CNN_DQN_QAT, self).__init__()
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        a_bits = act_bits if act_bits is not None else 8

        # Quantize the Input Image
        self.input_q = ActQuant(a_bits, momentum)

        # Convolution Layers (Batch Norm Folded)
        self.conv1 = QuantConvBN(copy.deepcopy(float_net.conv1), copy.deepcopy(float_net.bn1), weight_bits)
        self.act1 = ActQuant(a_bits, momentum)
        self.conv2 = QuantConvBN(copy.deepcopy(float_net.conv2), copy.deepcopy(float_net.bn2), weight_bits)
        self.act2 = ActQuant(a_bits, momentum)
        self.conv3 = QuantConvBN(copy.deepcopy(float_net.conv3), copy.deepcopy(float_net.bn3), weight_bits)
        self.act3 = ActQuant(a_bits, momentum)

        # Linear Layers
        self.head = QuantLinear(copy.deepcopy(float_net.head), weight_bits)
        self.act4 = ActQuant(a_bits, momentum)

        # The Output (Q-Values) is Left Unquantized, Regression Needs More Precision (Section 7)
        self.fc1 = QuantLinear(copy.deepcopy(float_net.fc1), weight_bits)

    # The Quantizer Sits After the ReLU so the Two Fuse at Inference (Section 3.2.1)
    def forward(self, x):
        x = x.to(self.conv1.conv.weight.device)
        x = self.input_q(x)

        x = self.act1(F.relu(self.conv1(x)))
        x = self.act2(F.relu(self.conv2(x)))
        x = self.act3(F.relu(self.conv3(x)))

        x = self.act4(F.relu(self.head(x.reshape(x.size(0), -1))))
        x = self.fc1(x)
        return x

# Switch Quantization and Batch Norm Freezing On or Off for a Network
def set_qat_state(net, quant_enabled, bn_frozen):
    for module in net.modules():
        if isinstance(module, WeightQuant):
            module.enabled = quant_enabled
        elif isinstance(module, ActQuant):
            module.enabled = quant_enabled and net.act_bits is not None
        elif isinstance(module, QuantConvBN):
            module.bn_frozen = bn_frozen

#============================================================================
# Export to Integer Weights
#============================================================================
# Pack Two 4-Bit Weights Into One Byte (8-Bit Weights are Stored as Is)
def pack_weights(wq, bits):
    flat = wq.flatten().to(torch.uint8)

    if bits == 4:
        if flat.numel() % 2 == 1:
            flat = torch.cat([flat, flat.new_zeros(1)])
        flat = (flat[0::2] << 4) | flat[1::2]

    return flat

def unpack_weights(packed, bits, shape):
    if bits == 4:
        flat = torch.stack([packed >> 4, packed & 0x0F], dim=1).flatten()
    else:
        flat = packed

    numel = int(np.prod(shape))
    return flat[:numel].reshape(shape)

# Export Integer Weights, Scales and Zero-Points for Every Layer
def export_integer_model(qnet):
    layers = [qnet.conv1, qnet.conv2, qnet.conv3, qnet.head, qnet.fc1]
    in_quants = [qnet.input_q, qnet.act1, qnet.act2, qnet.act3, qnet.act4]
    out_quants = [qnet.act1, qnet.act2, qnet.act3, qnet.act4, None]

    integer_acts = qnet.act_bits is not None
    n_w = 2 ** qnet.weight_bits

    exported = {
        "weight_bits": qnet.weight_bits,
        "act_bits": qnet.act_bits,
        "layers": [],
    }

    with torch.no_grad():
        if integer_acts:
            delta_in, z_in = qnet.input_q.get_qparams()
            exported["input_delta"] = float(delta_in)
            exported["input_z"] = int(z_in)

        for layer, q_in, q_out in zip(layers, in_quants, out_quants):
            w, b = layer.folded_params()

            # Per-Channel Weight Parameters (Same as During Training)
            delta_w, z_w = layer.weight_quant.get_qparams(w)
            wq = torch.clamp(torch.round(w / delta_w) + z_w, 0, n_w - 1)

            entry = {
                "kind": "conv" if isinstance(layer, QuantConvBN) else "linear",
                "weight": pack_weights(wq.cpu(), qnet.weight_bits),
                "shape": list(w.shape),
                "delta_w": delta_w.flatten().float().cpu(),
                "z_w": z_w.flatten().to(torch.uint8).cpu(),
            }

            if isinstance(layer, QuantConvBN):
                entry["stride"] = list(layer.conv.stride)
                entry["padding"] = list(layer.conv.padding)

            if integer_acts:
                delta_x, z_x = q_in.get_qparams()
                entry["delta_x"] = float(delta_x)
                entry["z_x"] = int(z_x)

                # 32-Bit Integer Bias with Scale delta_x * delta_w
                entry["bias"] = torch.round(b / (delta_x * delta_w.flatten())).to(torch.int32).cpu()

                if q_out is not None:
                    delta_y, z_y = q_out.get_qparams()
                    entry["delta_y"] = float(delta_y)
                    entry["z_y"] = int(z_y)
            else:
                entry["bias"] = b.float().cpu()

            exported["layers"].append(entry)

    return exported

#============================================================================
# Inference From the Exported Integer Model
#============================================================================
# Integer Arithmetic is Emulated with Integer-Valued float64 Tensors,
# which is Exact for these Layer Sizes (Eq. 4)
class IntegerDQN(nn.Module):

    def __init__(self, exported):
        super(IntegerDQN, self).__init__()
        self.weight_bits = exported["weight_bits"]
        self.act_bits = exported["act_bits"]
        self.exported = exported
        self.layers = []

        for entry in exported["layers"]:
            wq = unpack_weights(entry["weight"], self.weight_bits, entry["shape"]).double()
            shape = [-1] + [1] * (wq.dim() - 1)
            z_w = entry["z_w"].double().view(shape)
            delta_w = entry["delta_w"].double()

            layer = {
                "kind": entry["kind"],
                "delta_w": delta_w,
                "bias": entry["bias"],
            }

            if entry["kind"] == "conv":
                layer["stride"] = tuple(entry["stride"])
                layer["padding"] = tuple(entry["padding"])

            if self.act_bits is not None:
                # (wq - z_w) is the Integer Weight with the Zero-Point Removed
                layer["w_int"] = wq - z_w
                layer["delta_x"] = entry["delta_x"]
                layer["z_x"] = entry["z_x"]
                layer["delta_y"] = entry.get("delta_y")
                layer["z_y"] = entry.get("z_y")
            else:
                layer["w_deq"] = ((wq - z_w) * delta_w.view(shape)).float()

            self.layers.append(layer)

    def forward(self, x):
        x = x.detach().cpu()

        if self.act_bits is None:
            return self.forward_weight_only(x)

        n_levels = 2 ** self.act_bits

        # Quantize the Input Image
        x = torch.clamp(
            torch.round(x.float() / self.exported["input_delta"]) + self.exported["input_z"],
            0, n_levels - 1).double()

        for i, layer in enumerate(self.layers):
            last = (i == len(self.layers) - 1)

            # Remove the Input Zero-Point
            if layer["kind"] == "linear":
                x = x.reshape(x.size(0), -1)
            x_int = x - layer["z_x"]

            # Integer Accumulation (32-Bit) Plus the Integer Bias
            if layer["kind"] == "conv":
                acc = F.conv2d(x_int, layer["w_int"], None, layer["stride"], layer["padding"])
                acc = acc + layer["bias"].double().view(1, -1, 1, 1)
                ch_shape = (1, -1, 1, 1)
            else:
                acc = F.linear(x_int, layer["w_int"], None)
                acc = acc + layer["bias"].double().view(1, -1)
                ch_shape = (1, -1)

            # Last Layer: Dequantize to the Float Q-Values
            if last:
                scale = layer["delta_x"] * layer["delta_w"]
                return (acc * scale.view(ch_shape)).float()

            # Requantize to the Next Layer's Scale, Lower Clamp at z_y is the Fused ReLU
            multiplier = layer["delta_x"] * layer["delta_w"] / layer["delta_y"]
            y = torch.round(acc * multiplier.view(ch_shape)) + layer["z_y"]
            x = torch.clamp(y, layer["z_y"], n_levels - 1)

    # Weight-Only: Integer Weights, Float Activations
    def forward_weight_only(self, x):
        x = x.float()

        for i, layer in enumerate(self.layers):
            last = (i == len(self.layers) - 1)

            if layer["kind"] == "conv":
                x = F.conv2d(x, layer["w_deq"], layer["bias"], layer["stride"], layer["padding"])
            else:
                x = F.linear(x.reshape(x.size(0), -1), layer["w_deq"], layer["bias"])

            if not last:
                x = F.relu(x)

        return x

#============================================================================
# Quantization Aware Training
#============================================================================
def quantize_models(pnet, weight_bits, act_bits, config_name):

    # Reset Steps Done
    image_dqn.steps_done = 0

    # Build the QAT Policy Network From the Trained Network
    qat_policy_net = CNN_DQN_QAT(pnet, weight_bits, act_bits, act_momentum).to(device)

    # Create a Copy for the Target Network
    qat_target_net = copy.deepcopy(qat_policy_net)

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

    # Adam Optimiser
    qat_optimiser = optim.Adam(qat_policy_net.parameters(), alpha)

    # Training Phases: (Name, Episodes, Quantization On, Batch Norm Frozen)
    phases = [
        ("Float Warm-Up (Ranges Calibrating)", quant_delay_episodes, False, False),
        ("QAT (Batch Norm Batch Statistics)", bn_freeze_episode - quant_delay_episodes, True, False),
        ("QAT (Batch Norm Frozen)", qat_episodes - bn_freeze_episode, True, True),
    ]

    #============================================================================
    # Quantization Aware Training - Fine Tuning
    #============================================================================
    print(f'\nStart Quantization Aware Training for {config_name} (Fine-Tuning)...')
    start_time = time.time()

    # SetUp Tensorboard
    writer = SummaryWriter(log_dir=f"runs/QAT_{config_name}")

    # Train Each Phase
    for phase_name, phase_episodes, quant_on, bn_frozen in phases:

        if phase_episodes <= 0:
            continue

        print(f'Phase: {phase_name} for {phase_episodes} episodes...')

        # Both Networks Must Share the Same State
        set_qat_state(qat_policy_net, quant_on, bn_frozen)
        set_qat_state(qat_target_net, quant_on, bn_frozen)

        qat_policy_net.train()
        mean_reward, success_rate, mean_steps = deep_q_learning(
            env, train_param, explore_param, qat_optimiser, qat_policy_net,
            qat_target_net, episodes=phase_episodes, writer=writer, memory=qat_memory
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
    # Export QAT Model
    #============================================================================
    # Final Network Uses Quantization and the Long-Term Batch Norm Statistics
    set_qat_state(qat_policy_net, True, True)
    qat_policy_net.eval()
    qat_policy_net.cpu()

    # Export the Integer Weights, Scales and Zero-Points
    exported = export_integer_model(qat_policy_net)

    # Save the Trained Model
    output_file = f"{config_name}_QAT.pth"
    torch.save(exported, output_file)

    # Rebuild the Integer Network From the Saved File
    qat_quantized_net = IntegerDQN(torch.load(output_file, map_location="cpu"))
    qat_quantized_net.eval()

    #============================================================================
    # Check the Integer Model Matches the Simulated Quantization
    #============================================================================
    # Random Inputs Spread Over the Range of the Input Quantizer
    lo = qat_policy_net.input_q.min_val
    hi = qat_policy_net.input_q.max_val
    check_input = torch.rand(64, 4, screen_height, screen_width) * (hi - lo) + lo

    with torch.inference_mode():
        sim_out = qat_policy_net(check_input)
        int_out = qat_quantized_net(check_input)

    max_diff = (sim_out - int_out).abs().max().item()
    agreement = (sim_out.argmax(dim=1) == int_out.argmax(dim=1)).float().mean().item() * 100

    print(f'Simulated vs Integer Model: Max Q-Value Difference {max_diff:.6f} | '
          f'Same Action {agreement:.1f}%')

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

    baseline_results = benchmark_inference(
        fp32_model,
        input_tensor
    )

    # NOTE: The Integer Model Emulates Integer Arithmetic in float64 on a PC,
    # so this is NOT a Hardware Speedup. Real Gains Need an Integer Kernel on the MCU
    qat_results = benchmark_inference(
        qat_quantized_net,
        input_tensor
    )

    for name, results in [
        ("Baseline", baseline_results),
        (f"QAT for {config_name} (Emulated)", qat_results)
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

    # Weight Bits, Activation Bits (None = Weight-Only)
    models = {
        "W8A8": (8, 8),
        "W4A8": (4, 8),
        "W8WO": (8, None),
    }

    for name, (weight_bits, act_bits) in models.items():
        try:
            quantize_models(policy_net, weight_bits, act_bits, name)
        except (AssertionError, RuntimeError) as e:
            print(f"\nSkipping {name}: {e}")

    #============================================================================
    # Close Environment
    #============================================================================
    env.close()