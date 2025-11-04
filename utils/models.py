import torch 
import math

#################################################################################################
#
#   SIMPLE CNN Model
#
#################################################################################################

class SimpleModel(torch.nn.Module):
    def __init__(self, num_steps=1000,):
        super(SimpleModel,  self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 1, 3, padding=1)
        self.linear = torch.nn.Linear(1, 16)

    def forward(self, x, t):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        # t = t.to(torch.float32)
        t = t.view(-1, 1).to(torch.float32)
        t = self.linear(t)
        x = x + t.view(-1, 16, 1, 1)
        x = self.conv2(x)
        return x

#################################################################################################
#
#   BASIC UNET Model
#
#################################################################################################


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super(ConvBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.proj = torch.nn.Linear(embed_dim, out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x, embedding):
        x = self.conv1(x)
        emb_proj = self.proj(embedding).view(-1, x.size(1), 1, 1)
        x = torch.nn.functional.relu(x + emb_proj)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        return x
    
class UNetModel(torch.nn.Module):
    def __init__(self, num_steps=1000, embed_dim=16):
        super(UNetModel, self).__init__()

        self.embed = torch.nn.Embedding(num_steps, embed_dim)

        self.enc1 = ConvBlock(1, 16, embed_dim)
        self.enc2 = ConvBlock(16, 32, embed_dim)
        self.bottleneck = ConvBlock(32, 64, embed_dim)
        self.upconv2 = torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32, embed_dim)
        self.upconv1 = torch.nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32, 16, embed_dim)
        self.final = torch.nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x, t):
        emb = self.embed(t)
        enc1 = self.enc1(x, emb)
        enc2 = self.enc2(torch.nn.functional.max_pool2d(enc1, 2), emb)
        bottleneck = self.bottleneck(torch.nn.functional.max_pool2d(enc2, 2), emb)
        dec2 = self.dec2(torch.cat([enc2, self.upconv2(bottleneck)], 1), emb)
        dec1 = self.dec1(torch.cat([enc1, self.upconv1(dec2)], 1), emb)
        out = self.final(dec1)
        return out
    
#################################################################################################
#
#   ADVANCED UNET Model
#
#################################################################################################
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create a matrix to hold the positional encodings
        pe = torch.zeros(max_len, d_model)

        # Compute the positional encodings
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register pe as a buffer to avoid updating it during backpropagation
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Retrieve the positional encodings
        return self.pe[x]
    
class ResnetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, embed_channels):
        super(ResnetBlock, self).__init__()
        self.in_layers = torch.nn.Sequential(
            torch.nn.GroupNorm(16, in_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.emb_layers = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(embed_channels, out_channels)
        )
        self.out_layers = torch.nn.Sequential(
            torch.nn.GroupNorm(16, out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        if in_channels != out_channels:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x, embedding):
        _input = x
        x = self.in_layers(x)
        emb_out = self.emb_layers(embedding).view(-1, x.size(1), 1, 1)
        x = x + emb_out
        x = self.out_layers(x)
        return x + self.shortcut(_input)

class Upsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x

class Downsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class AdvUNetModel(torch.nn.Module):
    def __init__(self, num_steps=1000, embed_dim=64):
        super(AdvUNetModel, self).__init__()

        self.embed = torch.nn.Sequential(
            PositionalEncoding(embed_dim, num_steps),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, embed_dim),
        )

        self.conv_in = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.enc1_1 = ResnetBlock(16, 16, embed_dim)
        self.enc1_2 = ResnetBlock(16, 32, embed_dim)
        self.downconv1 = Downsample(32, 32)
        self.enc2_1 = ResnetBlock(32, 32, embed_dim)
        self.enc2_2 = ResnetBlock(32, 64, embed_dim)
        self.downconv2 = Downsample(64, 64)
        self.bottleneck_1 = ResnetBlock(64, 64, embed_dim)
        self.bottleneck_2 = ResnetBlock(64, 64, embed_dim)
        self.upconv2 = Upsample(64, 64)
        self.dec2_1 = ResnetBlock(128, 64, embed_dim)
        self.dec2_2 = ResnetBlock(64, 32, embed_dim)
        self.upconv1 = Upsample(32, 32)
        self.dec1_1 = ResnetBlock(64, 32, embed_dim)
        self.dec1_2 = ResnetBlock(32, 16, embed_dim)
        self.norm_out = torch.nn.GroupNorm(16, 16)
        self.conv_out = torch.nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x, t):
        emb = self.embed(t)

        x = self.conv_in(x)
        x = self.enc1_1(x, emb)
        enc1 = self.enc1_2(x, emb)
        x = self.downconv1(enc1)
        x = self.enc2_1(x, emb)
        enc2 = self.enc2_2(x, emb)
        x = self.downconv2(enc2)
        x = self.bottleneck_1(x, emb)
        x = self.bottleneck_2(x, emb)
        x = self.upconv2(x)
        x = torch.cat([x, enc2], 1)
        x = self.dec2_1(x, emb)
        x = self.dec2_2(x, emb)
        x = self.upconv1(x)
        x = torch.cat([x, enc1], 1)
        x = self.dec1_1(x, emb)
        x = self.dec1_2(x, emb)
        x = self.norm_out(x)
        x = torch.nn.functional.relu(x)
        x = self.conv_out(x)
        return x
