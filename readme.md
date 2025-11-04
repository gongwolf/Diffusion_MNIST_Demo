# PyTorch DDPM on MNIST

This project is a simple, clear implementation of a Denoising Diffusion Probabilistic Model (DDPM) in PyTorch, trained on the MNIST dataset.

It is based on the paper "Denoising Diffusion Probabilistic Models" by Ho et al. [[1]](https://www.google.com/search?q=%23ho2020denoisingdiffusionprobabilisticmodels).

## Table of Contents

  * [🚀 Installation](https://www.google.com/search?q=%23-installation)
  * [🏃 How to Run](https://www.google.com/search?q=%23-how-to-run)
  * [🧠 How It Works (The Theory)](https://www.google.com/search?q=%23-how-it-works-the-theory)
      * [1. The Forward Process (Noising)](https://www.google.com/search?q=%231-the-forward-process-noising)
      * [2. The Reverse Process (Denoising & Training)](https://www.google.com/search?q=%232-the-reverse-process-denoising--training)
      * [3. Sampling (Generation)](https://www.google.com/search?q=%233-sampling-generation)
  * [🗂️ File Structure](https://www.google.com/search?q=%23%EF%B8%8F-file-structure)
  * [References](https://www.google.com/search?q=%23references)

## 🚀 Installation

1.  Create a conda environment:
    ```bash
    conda create -n diffusion-env python=3.11
    ```
2.  Activate the environment:
    ```bash
    conda activate diffusion-env
    ```
3.  Install necessary packages:
    ```bash
    pip install torch torchvision torchaudio
    pip install matplotlib
    ```

## 🏃 How to Run

1.  **Download the Dataset:**
    Run the `0_mnist_dataset_download.py` script to download and prepare the MNIST dataset.

    ```bash
    python 0_mnist_dataset_download.py
    ```

2.  **(Optional) Visualize the Forward Process:**
    Open and run the `1_forward_process.ipynb` notebook. This is not required for training, but it provides an excellent visualization of the different noising schedulers defined in `scheduler.py`.

3.  **Run Training:**
    Execute the main training script. This will train the `AdvUNetModel` on MNIST and save the model weights to the `./data/` directory.

    ```bash
    python 4_mnist_diffusion.py
    ```

    You can edit this file to train `SimpleModel` or `UNetModel` instead.

4.  **Run Sampling (Inference):**
    *(Note: A dedicated `5_sample.py` script is not yet built, but the logic is implemented.)*

    To generate new images, you would:

      * Load the trained model weights (e.g., `mnist-model-AdvUNetModel.pth`).
      * Start with pure Gaussian noise: `x_T = torch.randn(...)`.
      * Iteratively denoise the image from `t = T-1` down to `0` using the `Alpha_NoiseScheduler.sample_prev_step()` method.

You're right, that section is very heavy on the math and can be tough to read in a `README`. It's better to lead with the *intuition* first and then connect it to the code.

Here's a refined version of that section, focusing on clarity and a more intuitive flow.

---
... (previous sections) ...

## 🧠 How It Works (The Theory)

### 1. The Forward Process (Noising)

The forward process is how we "break" the image.

> **The Idea:** Imagine taking a clear photo (`x_0`) and adding a tiny bit of TV static (Gaussian noise) to it. Now, do that again... and again... for many steps (`T`). By the end, the original photo is completely lost in the static, leaving only pure noise (`x_T`).

This is a **Markov process**, where each step `t` only depends on the step before it `t-1`.

A key trick from the DDPM paper is that we don't need to do this one step at a time. There's a mathematical shortcut that lets us jump *directly* from the clean image `x_0` to *any* noisy step `x_t` in a single calculation.

This "direct sampling" formula is what we use for training:

`x_t = sqrt(alpha_bar[t]) * x_0 + sqrt(1 - alpha_bar[t]) * ε`

* `x_0` is our original, clean image.
* `ε` is the random noise.
* `alpha_bar[t]` is a pre-calculated value (called the cumulative product of variance) that tells us *exactly* how much of the original image and how much noise to mix together for a given timestep `t`.

**💻 In this code:** This direct-sampling formula is implemented in **`scheduler.py`** inside `Alpha_NoiseScheduler.add_noise()`. The main **`4_mnist_diffusion.py`** script calls this function in every training step to create a noisy `x_t` sample.

---

### 2. The Reverse Process (Denoising & Training)

The reverse process is how we *create* an image.

> **The Idea:** We start with a screen full of pure static (`x_T`) and train a model to "guess" what the image looked like one step *less* noisy (`x_{T-1}`). We then take that slightly less noisy image, feed it back into the model, and ask it to guess `x_{T-2}`. We repeat this "denoising" guess all the way back to `t=0` to (hopefully) get a clean, new image.

**The Model's Job:**
Our model (the U-Net $\epsilon_{\theta}$) is not trained to guess the previous image directly. Instead, its job is much simpler:

It looks at a noisy image `x_t` and is trained to **predict the noise `ε` that was added to it.**

**The Loss Function:**
This makes training straightforward. We know the *actual* noise `ε` we added. We just need to teach the model to predict that *exact* noise.

The loss is simply the **Mean Squared Error (MSE)** between the *actual* noise and the *predicted* noise.

`Loss = MSE(ε, ε_θ)`

Or, more formally:
$$
L_{\text{simple}} := \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon - \epsilon_{\theta}(x_t, t) \right\|^2 \right]
$$

**The Training Loop:**
1.  Sample a clean image `x_0` from MNIST.
2.  Sample a random timestep `t`.
3.  Sample random noise `ε`.
4.  Create the noisy image `x_t` using the `add_noise()` shortcut.
5.  Feed `x_t` and `t` into our U-Net to get the `pred_noise = model(x_t, t)`.
6.  Calculate the loss: `loss = criterion(pred_noise, noise)`.
7.  Update the model.

**💻 In this code:** This entire loop is implemented in the `train()` function in **`4_mnist_diffusion.py`**. The U-Net models (`SimpleModel`, `UNetModel`, `AdvUNetModel`) are defined in **`models.py`**.

---

### 3\. Sampling (Generation)

Once the model $\epsilon_{\theta}$ is trained, we can use it to generate new images.

1. Sample $x_T$ from the normal distribution $\mathcal{N}(0, 1)$
2. For $t = T, \cdots, 1$:
   1. Sample $z$ from the normal distribution $\mathcal{N}(0, 1)$ if $t > 1$, else $z = 0$
   2. Compute the mean $\mu_t$ and variance $\sigma^2_t$, using $x_t$ and our schedule vars ($\bar{\alpha}_t$, $\beta_t$, etc.)
   3. Compute the previous stepL $x_{t-1} = \mu_t + \sigma_t * z$
4. Return $x_0$

The formula for calculating the mean of the previous step `x_{t-1}` is:

$$
\mu_{\theta}(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_{\theta}(x_t, t) \right)
$$

$$
\sigma^2_{\theta}(x_t, t) = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t
$$

**💻 In this code:** This sampling logic is implemented in **`scheduler.py`** as the **`Alpha_NoiseScheduler.sample_prev_step()`** method.

-----

## 🗂️ File Structure

```
.
│
├── 0_mnist_dataset_download.py    # 📥 Script to download MNIST data
├── 1_forward_process.ipynb        # 📊 Notebook to visualize the forward (noising) process
├── 2_reverse_process.ipynb        # 📝 Scratchpad notebook for reverse process concepts
├── 4_mnist_diffusion.py           # 🚀 The main training script
│
├── utils/                         # 🛠️ Utility directory
│   ├── models.py                  # 🤖 Model architectures (UNet, etc.)
│   ├── scheduler.py               # 🌊 Noise schedulers
│   └── tools.py                   # ⚙️ Helper functions (normalize, denormalize)
│
└── data/                          # 📁 Directory for datasets and saved model weights
```

-----

## 🧠 Core Components

This project implements and compares several key components for building a diffusion model.

### 🌊 Noise Schedulers (from `utils/scheduler.py`)

This file contains different methods for applying noise during the forward process, which are visualized in `1_forward_process.ipynb`.

1.  **`Random_NoiseScheduler`**: A simple, fixed-noise approach (`a*x + b*e`).
2.  **`SingleBeta_NoiseScheduler`**: Applies noise using a single, fixed beta value.
3.  **`Beta_NoiseScheduler`**: Applies noise from `x0` based on a linear beta schedule (`beta[t]`).
4.  **`Alpha_NoiseScheduler`**: The standard **DDPM scheduler** that uses `alpha_bar` for direct, cumulative noising from `x0` to `xt`. This is the one used for training in `4_mnist_diffusion.py`.

### 🤖 Model Architectures (from `utils/models.py`)

You've implemented three different models to predict the added noise `e`.

1.  **`SimpleModel`**: A basic CNN with a linear layer to embed the time `t`. A good starting point but lacks the power to capture complex details.
2.  **`UNetModel`**: A standard U-Net architecture.  This is the foundational model for diffusion, using:
      * Skip connections (`torch.cat`) to preserve low-level features.
      * `torch.nn.Embedding` for timesteps.
3.  **`AdvUNetModel`**: A more powerful, modern U-Net (used by default in your training script) which includes:
      * **Resnet Blocks**: Uses residual connections within its layers for more stable training.
      * **Positional Encoding**: Uses sinusoidal time embeddings (like in Transformers) for a richer representation of the timestep `t`.
      * **Group Normalization**: Uses `GroupNorm` instead of `BatchNorm`, which works better with small batch sizes.


## Acknowledgements

This project was built for educational purposes and was heavily inspired by the clear and concise `diffusion-from-scratch` repository by LambdaLabsML.

* **[LambdaLabsML/diffusion-from-scratch](https://github.com/LambdaLabsML/diffusion-from-scratch)**: Thank you for providing such an excellent, minimal implementation to learn from.

## References

1.  \<a id="ho2020denoisingdiffusionprobabilisticmodels"\>\</a\> Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239).