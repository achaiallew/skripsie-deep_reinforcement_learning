import os, shutil
import torch, numpy as np

PTH_FILE = "W8WO_QAT.pth"
OUT_FILE = "W8WO_QAT_weights.h"

if os.path.exists(OUT_FILE):
    shutil.copy(OUT_FILE, "model_weights_old.h")
    print("Backed up old header to model_weights_old.h")

sd = torch.load(PTH_FILE, map_location="cpu", weights_only=False)
print("weight_bits:", sd["weight_bits"], " act_bits:", sd["act_bits"])

NAMES = ["conv1", "conv2", "conv3", "head", "fc1"]
EXPECTED = {"conv1": [16, 4, 5, 5], "conv2": [32, 16, 5, 5], "conv3": [32, 32, 5, 5],
            "head": [128, 512], "fc1": [3, 128]}

def write(f, ctype, name, a):
    a = a.flatten()
    f.write(f"const {ctype} {name}[{a.size}] = {{\n")
    for i in range(0, a.size, 16):
        chunk = a[i:i+16]
        f.write(",".join(f"{x:.9g}f" if ctype == "float" else str(int(x)) for x in chunk) + ",\n")
    f.write("};\n\n")

with open(OUT_FILE, "w") as f:
    for name, layer in zip(NAMES, sd["layers"]):
        q     = layer["weight"].numpy().astype(np.uint8)
        delta = layer["delta_w"].detach().numpy().astype(np.float32)
        z     = layer["z_w"].numpy().astype(np.uint8)
        bias  = layer["bias"].detach().numpy().astype(np.float32)
        shape = list(layer["shape"])

        print(f"{name}: kind={layer['kind']}  shape={shape}  stride={layer.get('stride')}  padding={layer.get('padding')}")
        print(f"   q range {q.min()}..{q.max()}   z range {z.min()}..{z.max()}   delta range {delta.min():.6f}..{delta.max():.6f}")

        if shape != EXPECTED[name]:
            print(f"   WARNING: expected shape {EXPECTED[name]}")
        if int(np.prod(shape)) != q.size:
            print("   WARNING: weight count does not match shape")
        pad = layer.get("padding")
        if pad is not None and any(int(p) != 0 for p in pad):
            print("   WARNING: padding is not zero. The C code assumes no padding.")

        write(f, "uint8_t", f"{name}_q", q)
        write(f, "float",   f"{name}_delta", delta)
        write(f, "uint8_t", f"{name}_z", z)
        write(f, "float",   f"{name}_b", bias)

print(f"Wrote {OUT_FILE}")