import math
import torch 

class Random_NoiseScheduler:
    """
    Wrapper for your add_random_noise function.
    Applies a fixed a*x + b*noise. The time step 't' is ignored.
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b
        
    def add_noise(self, x0, t, noise=None):
        if noise ==  None:
            noise = torch.randn_like(x0)
        
        # 't' is ignored, but we return a different noise sample each time
        return self.a * x0 + self.b * noise

class SingleBeta_NoiseScheduler:
    """
    Wrapper for your add_single_beta_noise function.
    Applies noise with a single, fixed beta. The time step 't' is ignored.
    """
    def __init__(self, beta):
        self.beta = beta
        self.sqrt_1_beta = math.sqrt(1 - beta)
        self.sqrt_beta = math.sqrt(beta)

    def add_noise(self, x0, t, noise=None):
        if noise==None:
            noise = torch.randn_like(x0)
        return self.sqrt_1_beta * x0 + self.sqrt_beta * noise

class Beta_NoiseScheduler:
    """
    Your Beta_NoiseScheduler.
    Applies noise to x0 based on beta[t]. This is NOT a cumulative
    diffusion process, but rather x0 + one_step_noise(t).
    """
    def __init__(self, steps=24, beta_start=1e-4, beta_end=0.6, device='cpu'):
        super(Beta_NoiseScheduler, self).__init__()
        self.steps = steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = torch.linspace(beta_start, beta_end, steps).to(device)

    def add_noise(self, x0, t, noise=None):
        """
        Adds a single step of noise (using beta[t]) to the *original* image x0.
        """
        if noise==None:
            noise = torch.randn_like(x0)        

        # t is a batch of indices, so beta is a batch of values (B,)
        beta = self.beta[t]
        # --- Reshape for broadcasting (B,) -> (B, 1, 1, 1) ---
        beta = beta.view(-1, 1, 1, 1) 
        # --- Use torch.sqrt for tensor math, not math.sqrt ---
        return torch.sqrt(1 - beta) * x0 + torch.sqrt(beta) * noise
    
class Alpha_NoiseScheduler:
    """
    Your Alpha_NoiseScheduler.
    This is the standard DDPM direct forward process.
    """
    def __init__(self, steps=24, beta_start=1e-4, beta_end=0.6, device='cpu'):
        super(Alpha_NoiseScheduler, self).__init__()
        self.steps = steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = torch.linspace(beta_start, beta_end, steps).to(device)
        self.alpha = (1. - self.beta).to(device)
        self.alpha_bar = torch.cumprod(self.alpha, 0).to(device)
        self.var = ((1 - self.alpha_bar[:-1]) / (1 - self.alpha_bar[1:])) * self.beta[1:]

    def add_noise(self, x0, t, noise=None):
        """
        Adds arbitrary noise to an image (direct sampling).
        """
        if noise==None:
            noise = torch.randn_like(x0)

        # t is a batch of indices, so alpha_bar is a batch of values (B,)
        alpha_bar = self.alpha_bar[t]
        # --- Reshape for broadcasting (B,) -> (B, 1, 1, 1) ---
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)
        # --- Use torch.sqrt for tensor math, not math.sqrt ---
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
    
    def sample_prev_step(self, xt, t, pred_noise):
       z = torch.randn_like(xt)
       t = t.view(-1, 1, 1, 1)
       z[t.expand_as(z) == 0] = 0
       mean = (1 / torch.sqrt(self.alpha[t])) * (xt - (self.beta[t] / torch.sqrt(1 - self.alpha_bar[t])) * pred_noise)
       var = ((1 - self.alpha_bar[t - 1])  / (1 - self.alpha_bar[t])) * self.beta[t]
       sigma = torch.sqrt(var)
       x = mean + sigma * z
       return x