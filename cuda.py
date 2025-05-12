import torch, platform

print("Torch:", torch.__version__, "Python:", platform.python_version())
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    count = torch.cuda.device_count()
    [print(f"[{idx}] {torch.cuda.get_device_name(idx)}") for idx in range(count)]
