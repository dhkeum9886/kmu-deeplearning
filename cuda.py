import torch


# print("Torch:", torch.__version__, "Python:", platform.python_version())
# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     count = torch.cuda.device_count()
#     [print(f"[{idx}] {torch.cuda.get_device_name(idx)}") for idx in range(count)]


def get_device() -> tuple[torch.device, int]:
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        [print(f"[{idx}] {torch.cuda.get_device_name(idx)}") for idx in range(count)]
        return torch.device("cuda"), torch.cuda.device_count()
    else:
        print("GPU 미감지: CPU로 실행")
        return torch.device("cpu"), 0
