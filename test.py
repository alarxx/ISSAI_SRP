import os
import dotenv 
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
dotenv.load_dotenv()  # загрузит переменные из .env

import torch

if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_devices}")
    for i in range(torch.cuda.device_count()):
        print(f"Logical device {i} is physical GPU {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")