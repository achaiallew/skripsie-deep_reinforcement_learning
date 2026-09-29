#!/usr/bin/env python3
"""
embedded_export.py

Convert the existing TorchAO QAT-converted checkpoint into an ExecuTorch
.PTE file suitable for the Raspberry Pi Pico 2 portable-operator workflow.

IMPORTANT:
    This script does NOT retrain and does NOT re-quantize the model.

    Your existing dqn_qat_quantized.pth contains TorchAO low-bit tensor
    subclasses (INT4 weights + INT8 dynamic activations). The current
    official ExecuTorch Cortex-M/Pico 2 backend does not yet integrate the
    4w8a path used by that checkpoint. Therefore, this script dequantizes
    the learned QAT weights back to ordinary FP32 tensors and exports the
    resulting model as an FP32 .pte.

    The learned QAT weights are retained; only their storage/execution
    representation is changed for deployment.

Usage:
    python embedded_export.py

Optional:
    python embedded_export.py --checkpoint dqn_qat_quantized.pth
    python embedded_export.py --output dqn_qat_pico2_fp32.pte
    python embedded_export.py --skip-fusion

After this script succeeds, from your ExecuTorch repository root:

    ./examples/raspberry_pi/pico2/build_firmware_pico.sh \
        --model=/path/to/dqn_qat_pico2_fp32.pte

Do NOT use --cmsis with this FP32 PTE.

Requirements:
    - PyTorch
    - TorchAO
    - ExecuTorch
    - image_dqn.py in the same directory (or importable from PYTHONPATH)
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_ACTIONS = 3
SCREEN_HEIGHT = 56
SCREEN_WIDTH = 56
HIDDEN_LAYER_SIZE = (128,)

DEFAULT_CHECKPOINT = "dqn_qat_quantized.pth"
DEFAULT_OUTPUT = "dqn_qat_pico2_fp32.pte"

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Import the exact network used during training.
#
# image_dqn.py has a module-global "device" variable and CNN_DQN.forward()
# moves the input to that device. Since this script is for CPU/embedded
# export, force the imported module to CPU.
# ---------------------------------------------------------------------------

try:
    import image_dqn

    image_dqn.device = DEVICE
    CNN_DQN = image_dqn.CNN_DQN
except Exception as exc:
    raise RuntimeError(
        "Could not import CNN_DQN from image_dqn.py. "
        "Make sure image_dqn.py is in the same directory as this script."
    ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    """Load the state_dict saved by image_qat(1).py."""
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Place '{path.name}' beside this script or pass --checkpoint."
        )

    print(f"Loading checkpoint: {path}")

    # This checkpoint was produced by our own image_qat(1).py script using
    # torchao tensor subclasses. PyTorch 2.6+ defaults torch.load() to
    # weights_only=True, which rejects the TorchAO IntxUnpackedToInt8Tensor
    # class unless it is explicitly allow-listed.
    #
    # We use weights_only=False here ONLY because this is the trusted local
    # checkpoint created by your own training/QAT script. Never do this for
    # a checkpoint from an untrusted source.
    try:
        state_dict = torch.load(
            path,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions that do not accept the
        # weights_only keyword.
        state_dict = torch.load(
            path,
            map_location=DEVICE,
        )

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            "The checkpoint is not a state_dict dictionary. "
            "Expected the output of torch.save(model.state_dict(), ...)."
        )

    return state_dict


def materialize_float_tensor(
    tensor: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """
    Convert a TorchAO quantized tensor subclass into a normal FP32 tensor.

    Ordinary floating-point tensors are simply copied.
    TorchAO low-bit tensors are dequantized using their own scale/zero-point
    metadata. No new quantization is performed here.
    """
    if torch.is_floating_point(tensor):
        return tensor.detach().to(
            device=DEVICE,
            dtype=torch.float32,
        ).clone()

    # A quantized TorchAO tensor subclass normally provides dequantize().
    dequantize_fn = getattr(tensor, "dequantize", None)

    if dequantize_fn is None:
        raise RuntimeError(
            f"Cannot dequantize checkpoint tensor '{name}'. "
            f"Tensor type: {type(tensor)}, dtype: {tensor.dtype}"
        )

    try:
        result = dequantize_fn()
    except Exception as exc:
        raise RuntimeError(
            f"TorchAO dequantization failed for '{name}'. "
            f"Tensor type: {type(tensor)}, dtype: {tensor.dtype}"
        ) from exc

    if not torch.is_floating_point(result):
        result = result.float()

    return result.detach().to(
        device=DEVICE,
        dtype=torch.float32,
    ).clone()


def build_fp32_model(
    state_dict: dict[str, torch.Tensor],
) -> CNN_DQN:
    """
    Reconstruct the original CNN_DQN and load the learned QAT weights after
    materializing any quantized TorchAO parameters as ordinary FP32 tensors.
    """
    print("Creating FP32 CNN_DQN architecture...")

    model = CNN_DQN(
        SCREEN_HEIGHT,
        SCREEN_WIDTH,
        NUM_ACTIONS,
        HIDDEN_LAYER_SIZE,
    ).to(DEVICE)

    float_state_dict: dict[str, torch.Tensor] = {}

    quantized_names: list[str] = []

    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(
                f"Checkpoint entry '{name}' is not a tensor: {type(value)}"
            )

        # Parameters that are quantized in this checkpoint are expected to be
        # low-bit TorchAO tensor subclasses. Other parameters/buffers (conv
        # weights, BN statistics, biases, num_batches_tracked, etc.) remain in
        # their original representation.
        if (
            name.endswith(".weight")
            and not torch.is_floating_point(value)
        ):
            quantized_names.append(name)
            float_state_dict[name] = materialize_float_tensor(value, name)
        else:
            # Preserve integer buffers such as num_batches_tracked.
            float_state_dict[name] = value.detach().cpu().clone()

    print()
    print("Quantized tensors dequantized:")
    if quantized_names:
        for name in quantized_names:
            print(f"  - {name}")
    else:
        print("  (none detected)")

    missing, unexpected = model.load_state_dict(
        float_state_dict,
        strict=True,
    )

    if missing or unexpected:
        raise RuntimeError(
            f"State-dict mismatch after dequantization.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}"
        )

    # Inference only.
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


def fuse_batchnorm(model: CNN_DQN) -> CNN_DQN:
    """
    Fuse Conv2d + BatchNorm2d pairs for a smaller/simpler inference graph.

    This is not quantization and does not retrain the model.
    """
    model = copy.deepcopy(model).eval()

    print("Fusing Conv2d + BatchNorm2d layers...")

    pairs = [
        ("conv1", "bn1"),
        ("conv2", "bn2"),
        ("conv3", "bn3"),
    ]

    for conv_name, bn_name in pairs:
        conv = getattr(model, conv_name)
        bn = getattr(model, bn_name)

        fused_conv = fuse_conv_bn_eval(conv, bn)

        setattr(model, conv_name, fused_conv)
        setattr(model, bn_name, nn.Identity())

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    model.eval()

    return model


class ExportWrapper(nn.Module):
    """
    Export-only wrapper.

    We intentionally do not call CNN_DQN.forward() here because the original
    training model contains x.to(device), which is unnecessary for a fixed
    embedded CPU graph and can introduce device-specific export behavior.
    """

    def __init__(self, model: CNN_DQN):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN blocks
        x = F.relu(self.model.bn1(self.model.conv1(x)))
        x = F.relu(self.model.bn2(self.model.conv2(x)))
        x = F.relu(self.model.bn3(self.model.conv3(x)))

        # Flatten + fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.model.head(x))
        x = self.model.fc1(x)

        return x


def compare_models(
    quantized_model: nn.Module,
    fp32_model: nn.Module,
) -> None:
    """
    Compare the already-converted TorchAO model with the materialized FP32
    model on one deterministic test input.

    This checks that the dequantization step has not accidentally changed the
    network structure/weights beyond the expected numerical difference.
    """
    print()
    print("Running pre-export numerical check...")

    example_input = torch.linspace(
        0.0,
        1.0,
        SCREEN_HEIGHT * SCREEN_WIDTH * 4,
        dtype=torch.float32,
    ).reshape(
        1,
        4,
        SCREEN_HEIGHT,
        SCREEN_WIDTH,
    )

    quantized_model = quantized_model.to(DEVICE).eval()
    fp32_model = fp32_model.to(DEVICE).eval()

    with torch.inference_mode():
        y_quant = quantized_model(example_input)
        y_float = fp32_model(example_input)

    max_abs_error = (y_quant - y_float).abs().max().item()
    mean_abs_error = (y_quant - y_float).abs().mean().item()

    q_action = int(y_quant.argmax(dim=1).item())
    f_action = int(y_float.argmax(dim=1).item())

    print(f"  Quantized-model Q values: {y_quant.flatten().tolist()}")
    print(f"  FP32 Q values:           {y_float.flatten().tolist()}")
    print(f"  Max absolute difference: {max_abs_error:.8f}")
    print(f"  Mean absolute difference:{mean_abs_error:.8f}")
    print(f"  Greedy action (quant):   {q_action}")
    print(f"  Greedy action (FP32):    {f_action}")

    if q_action != f_action:
        print()
        print(
            "WARNING: dequantized FP32 model selected a different greedy "
            "action for this synthetic test input."
        )
        print(
            "This does not automatically mean deployment is invalid; test "
            "the model with representative states before flashing."
        )


def export_to_pte(
    model: nn.Module,
    output_path: Path,
) -> None:
    """Export the FP32 deployment graph and serialize it as a .pte file."""
    example_input = torch.randn(
        1,
        4,
        SCREEN_HEIGHT,
        SCREEN_WIDTH,
        dtype=torch.float32,
        device=DEVICE,
    )

    print()
    print("Running torch.export...")

    with torch.inference_mode():
        exported_program = torch.export.export(
            model,
            (example_input,),
            strict=True,
        )

    print("torch.export succeeded.")

    print("Lowering to ExecuTorch...")

    # No TorchAO quantized operators are present now, so the external
    # quantized AOT operator library is NOT required for this path.
    from executorch.exir import to_edge_transform_and_lower

    with torch.inference_mode():
        executorch_program = (
            to_edge_transform_and_lower(exported_program)
            .to_executorch()
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    executorch_program.save(str(output_path))

    print()
    print(f"ExecuTorch PTE written to:")
    print(f"  {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KiB")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an existing TorchAO QAT-converted DQN checkpoint "
            "to an FP32 ExecuTorch PTE for the Pico 2 portable path."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        type=Path,
        help=f"Input TorchAO checkpoint (default: {DEFAULT_CHECKPOINT})",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        type=Path,
        help=f"Output PTE file (default: {DEFAULT_OUTPUT})",
    )

    parser.add_argument(
        "--skip-fusion",
        action="store_true",
        help="Do not fuse Conv2d + BatchNorm2d before export.",
    )

    args = parser.parse_args()

    print("=" * 72)
    print("DQN -> ExecuTorch Pico 2 Export")
    print("=" * 72)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device:          {DEVICE}")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Output:          {args.output}")
    print()

    # -----------------------------------------------------------------------
    # 1. Load already-converted TorchAO state_dict
    # -----------------------------------------------------------------------
    quantized_state_dict = load_checkpoint(args.checkpoint)

    # -----------------------------------------------------------------------
    # 2. Load the exact quantized model representation for validation
    # -----------------------------------------------------------------------
    quantized_model = CNN_DQN(
        SCREEN_HEIGHT,
        SCREEN_WIDTH,
        NUM_ACTIONS,
        HIDDEN_LAYER_SIZE,
    ).to(DEVICE)

    print("Restoring TorchAO quantized tensors with assign=True...")

    quantized_model.load_state_dict(
        quantized_state_dict,
        assign=True,
    )

    quantized_model.eval()

    for parameter in quantized_model.parameters():
        parameter.requires_grad_(False)

    print("TorchAO checkpoint restored successfully.")

    # -----------------------------------------------------------------------
    # 3. Materialize QAT weights as ordinary FP32 tensors.
    #    This is dequantization, NOT re-quantization.
    # -----------------------------------------------------------------------
    fp32_model = build_fp32_model(quantized_state_dict)

    # -----------------------------------------------------------------------
    # 4. Validate that the learned QAT weights survived dequantization.
    # -----------------------------------------------------------------------
    compare_models(
        quantized_model,
        fp32_model,
    )

    # -----------------------------------------------------------------------
    # 5. Fuse Conv + BatchNorm for a simpler embedded graph.
    # -----------------------------------------------------------------------
    if not args.skip_fusion:
        fp32_model = fuse_batchnorm(fp32_model)

    # -----------------------------------------------------------------------
    # 6. Use an export-only wrapper so no x.to(device) appears in the graph.
    # -----------------------------------------------------------------------
    export_model = ExportWrapper(fp32_model).eval()

    for parameter in export_model.parameters():
        parameter.requires_grad_(False)

    # -----------------------------------------------------------------------
    # 7. Export + serialize to PTE.
    # -----------------------------------------------------------------------
    export_to_pte(
        export_model,
        args.output,
    )

    print()
    print("=" * 72)
    print("EXPORT COMPLETE")
    print("=" * 72)
    print()
    print("The generated file is an FP32 ExecuTorch PTE derived from your")
    print("already-trained QAT checkpoint. No retraining or re-quantization")
    print("was performed.")
    print()
    print("Pico 2 firmware command:")
    print()
    print(
        "  ./examples/raspberry_pi/pico2/build_firmware_pico.sh "
        f"--model={args.output}"
    )
    print()
    print("Do NOT add --cmsis for this FP32 PTE.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
