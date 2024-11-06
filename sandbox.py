import torch

# Assuming img1 and img2 are your two images of shape (32, 32, 3)
# Ensure they are PyTorch tensors
img1 = torch.randn(32, 32, 3)  # Example tensor
img2 = torch.randn(32, 32, 3)  # Example tensor

# Compute the difference mask
mask = (img1 != img2).any(dim=2)
mask_brute = torch.empty(32, 32, dtype=torch.bool)
for i in range(32):
    for j in range(32):
        mask_brute[i, j] = (img1[i, j] != img2[i, j]).any()
assert torch.all(mask == mask_brute)