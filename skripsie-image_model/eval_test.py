import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime

# Import the Main DQN
from image_dqn import CNN_DQN
# Check for GPU Availability
device = torch.device("cpu")

# Load Model
model_trained = "dqn_qat_quantized.pth" 

# ---- MODEL HYPERPARAMETERS ----
num_actions = 3                  
screen_height = 56
screen_width = 56

# Define the Hidden Layer Size
hiddenLayerSize = (128,)


# Create the DQN
qat_net = CNN_DQN(screen_height, screen_width, num_actions, hiddenLayerSize).to(device)

# Load the Trained Quantized Weights
state_dict = torch.load(model_trained, map_location=device)
qat_net.load_state_dict(state_dict, assign=True)

# This Model is for Inference only.
# Quantized integer parameters MUST NOT require gradients.
for param in qat_net.parameters():
    param.requires_grad_(False)

# Set to Evaluation Mode
qat_net.eval()

sample_inputs = (torch.randn(1, 4, screen_height, screen_width),)

et_program = to_edge_transform_and_lower(
    torch.export.export(qat_net, sample_inputs),
    partitioner=[XnnpackPartitioner()]
).to_executorch()

with open("dqn_qat.pte", "wb") as f:   # renamed from add.pte for sanity
    f.write(et_program.buffer)

runtime = Runtime.get()
runtime_program = runtime.load_program("add.pte")
method = runtime_program.load_method("forward")
output = method.execute(sample_inputs)[0]

torch.testing.assert_close(output, qat_net(*sample_inputs))
print("Output:", output)