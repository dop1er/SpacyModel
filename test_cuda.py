import torch
print(torch.cuda.is_available())       # должно вывести True
print(torch.cuda.device_count())       # >=1
print(torch.cuda.get_device_name(0))   # NVIDIA GeForce RTX 3070
