import torch

if torch.cuda.is_available():
    print("GPU is available!")
else:
    print("Please switch to a GPU-enabled environment.")