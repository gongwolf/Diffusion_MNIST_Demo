from utils.scheduler import Random_NoiseScheduler, SingleBeta_NoiseScheduler, Beta_NoiseScheduler, Alpha_NoiseScheduler
from utils.models import UNetModel, SimpleModel, AdvUNetModel

import torch 
import torchvision
import torchvision.transforms as transforms


def train(noise_scheduler, model, dataset, batch_size=128, epochs=80, lr=1e-3, device="cpu"):
    model = model(num_steps=noise_scheduler.steps).to(device)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Train
    for epoch in range(epochs):
        loss_epoch = 0
        n = 0
        for x, _ in data_loader:
            x = x.to(device)
            optimizer.zero_grad()
            
            # Generate random noise and time steps
            noise = torch.randn_like(x, device=device)
            t = torch.randint(0, noise_scheduler.steps, (x.size(0),), device=device)
            
            # Create noised image
            x_t = noise_scheduler.add_noise(x, t, noise)
            
            # Predict the noise
            pred_noise = model(x_t, t)
            
            # Calculate loss
            loss = criterion(pred_noise, noise)
            loss.backward()
            optimizer.step()
            
            loss_epoch += loss.item() * x.size(0) # Store total loss for epoch
            n += x.size(0)

        loss_epoch /= n
        print(f"Epoch {epoch}, Loss {loss_epoch:.4f}")

    torch.save(model.state_dict(), f'./data/mnist-model-{model.__class__.__name__}.pth')
    print("Training complete. Model saved to 'mnist-model.pth'")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    scheduler = Alpha_NoiseScheduler(steps=2000, beta_start=1e-4, beta_end=0.02, device=device)

    # train(scheduler, SimpleModel, train_dataset, device=device, epochs=1000)
    # train(scheduler, UNetModel, train_dataset,device=device, epochs=100)
    # train(scheduler, AdvUNetModel, train_dataset,device=device, epochs=100)

