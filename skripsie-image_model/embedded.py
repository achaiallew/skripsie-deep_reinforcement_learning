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
import torch.nn.functional as F

# Imports for ExecuTorch (converting model)
from executorch.exir import to_edge_transform_and_lower

# Check for GPU Availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import the Main DQN
from image_dqn import CNN_DQN

#============================================================================
# Settings
#============================================================================
# Load Model
model_trained = "dqn_qat_quantized.pth" 

# ---- MODEL HYPERPARAMETERS ----
num_actions = 3                  
screen_height = 56
screen_width = 56

# Define the Hidden Layer Size
hiddenLayerSize = (128,)

#============================================================================
# Load Original Model
#============================================================================
if exists(model_trained):
    # Create the DQN
    qat_net = CNN_DQN(screen_height, screen_width, num_actions, hiddenLayerSize).to(device)\
    
    # Load the Trained Quantized Weights
    state_dict = torch.load(model_trained, map_location=device)
    qat_net.load_state_dict(state_dict, assign=True)

    # This Model is for Inference only.
    # Quantized integer parameters MUST NOT require gradients.
    for param in qat_net.parameters():
        param.requires_grad_(False)

    # Set to Evaluation Mode
    qat_net.eval()
    print(f'Loaded trained model from {model_trained}')
else:
    raise FileNotFoundError(f'No saved model found at {model_trained}.')

#============================================================================
# Load ExecuTorch Quantized AOT Operators
#============================================================================

# CHANGE THIS TO YOUR ACTUAL LIBRARY LOCATION
quantized_ops_lib = r"C:\path\to\quantized_ops_aot_lib.dll"

print(f"Loading quantized AOT library:")
print(quantized_ops_lib)

torch.ops.load_library(quantized_ops_lib)

print("Quantized AOT library loaded successfully.")


#============================================================================
# Export the Model to Exportable Graph
#============================================================================
# Example Inputs (same shape as DQN)
example_input = torch.randn(1, 4, screen_height, screen_width)

# Export PyTorch Model
exportable_model = torch.export.export(qat_net, (example_input,))

#============================================================================
# Convert to an ExecuTorch Program
#============================================================================
executorch_program = to_edge_transform_and_lower(exportable_model).to_executorch()
executorch_program.save("dqn_qat_exportable.pte")

#============================================================================
# Build the Pico 2 Firmware
#============================================================================
# Move .pte file to Executorch\examples\raspberry pi \pico
# ./examples/raspberry_pi/pico2/build_firmware_pico.sh --model=dqn_trained.pte