import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.ToTensor(),
    # Normalize with mean 0.5 and standard deviation 0.5
    transforms.Normalize((0.5,), (0.5,)) 
])

# 2. Download and Load the Training Dataset
# 'root="./data"' specifies where to save the files.
# 'train=True' indicates the training set.
# 'download=True' automatically fetches the data if it's not present.
# 'transform=transform' applies the transformations defined above.
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

# 3. Download and Load the Test Dataset
test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

