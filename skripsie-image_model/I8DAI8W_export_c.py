import torch, numpy as np

sd = torch.load("I8DAI8W_QAT.pth", map_location="cpu", weights_only=False)

# ---- 1. Merge BatchNorm into conv ----
def fold(conv, bn):
    w  = sd[f"{conv}.weight"].numpy();  b  = sd[f"{conv}.bias"].numpy()
    g  = sd[f"{bn}.weight"].numpy();    be = sd[f"{bn}.bias"].numpy()
    m  = sd[f"{bn}.running_mean"].numpy(); v = sd[f"{bn}.running_var"].numpy()
    f = g / np.sqrt(v + 1e-5)
    return (w * f[:, None, None, None]).astype(np.float32), ((b - m) * f + be).astype(np.float32)

# ---- 2. Get the int8 layers ----
def get_int(name):
    t = sd[f"{name}.weight"]
    q = t.qdata.numpy().astype(np.int8)
    s = t.scale.numpy().reshape(-1).astype(np.float32)
    z = t.zero_point.numpy().reshape(-1).astype(np.int32)
    # ---- 3. Check ----
    rebuilt = (q.astype(np.float32) - z[:, None]) * s[:, None]
    err = np.abs(rebuilt - t.dequantize().numpy()).max()
    print(f"{name}: shape={q.shape}  max_error={err:.8f}")
    print(f"   zero_point all zero? {bool((z == 0).all())}")
    print(f"   activation_quantization: {t.activation_quantization}")
    print(f"   block_size: {t.block_size}   scale count: {s.size}")
    return q, s, z

# ---- 4. Write the C header ----
def write(f, ctype, name, a):
    a = a.flatten()
    f.write(f"const {ctype} {name}[{a.size}] = {{\n")
    for i in range(0, a.size, 16):
        chunk = a[i:i+16]
        f.write(",".join(f"{x:.9g}f" if ctype == "float" else str(int(x)) for x in chunk) + ",\n")
    f.write("};\n\n")

with open("model_weights.h", "w") as f:
    for i in (1, 2, 3):
        w, b = fold(f"conv{i}", f"bn{i}")
        write(f, "float", f"conv{i}_w", w)
        write(f, "float", f"conv{i}_b", b)
    for name in ("head", "fc1"):
        q, s, z = get_int(name)
        write(f, "int8_t", f"{name}_q", q)
        write(f, "float",  f"{name}_scale", s)
        write(f, "float",  f"{name}_bias", sd[f"{name}.bias"].numpy())
print("Wrote model_weights.h")