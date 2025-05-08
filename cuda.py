import torch, platform

print("Torch:", torch.__version__, "Python:", platform.python_version())
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU  개수:", torch.cuda.device_count())
    print("주 GPU :", torch.cuda.get_device_name(0))
