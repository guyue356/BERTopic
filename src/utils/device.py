import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[GPU] Available, using: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("[CPU] No GPU detected, using CPU.")
    return device


def get_device_info() -> dict:
    info = {
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }
    return info
