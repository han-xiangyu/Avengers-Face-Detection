import torch

# Check if CUDA is available in your PyTorch installation
print("CUDA available:", torch.cuda.is_available())

# If CUDA is available, you can also check the version
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
