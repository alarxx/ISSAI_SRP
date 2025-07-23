import torch

def print_available_devices():
    """
        Considers only CPU and NVIDIA devices
    """
    print("Available devices (CPU and CUDA devices):")
    print(" - CPU")
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"Number of CUDA devices: {num_devices}")
        for i in range(torch.cuda.device_count()):
            print(f" - CUDA:{i} ({torch.cuda.get_device_name(i)})")
    else:
        print("CUDA is not available.")
