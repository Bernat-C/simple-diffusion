# simple-diffusion
Diffusion project to learn the basics of conditional generation of images.

We are using a U-Net with Sinusoidal Position Embeddings defining the timestep in the diffusion process.

### The Sinusoidal Position Embedding: ###

The sinusoidal position embedding converts a scalar value $t$ (e.g., a timestep) into a high-dimensional vector. This provides the network with a rich, continuous representation of $t$, helping it understand the relationship between adjacent steps and learn smooth transitions.

1. <b>Define Dimensions</b>: Set the total embedding dimension $D$ and the half-dimension $d= D / 2$.

2. <b>Calculate Frequency Divisor ($\beta$)</b>: Calculate the scaling factor $\beta$ based on the half-dimension d.
   
$$\beta = \frac{log(10000)}{d−1​}$$

3. <b>Calculate Frequencies</b>: Create a vector of $d$ inverse frequencies. The $i$-th frequency (for $i$ from 0 to $d$−1) is:

$$inv\_freq_i ​= e^{−i · β}$$

4. <b>Scale Timesteps</b>: Multiply each timestep $t$ in the batch (shape [$B$]) by the inverse frequency vector (shape [$d$]). This broadcasted operation creates a scaledtime tensor of shape [$B$,$d$].

$$scaled\_time_{b,i}​ = t_b​ ⋅ inv\_freq_i​$$

5. <b>Apply Sine and Cosine</b>: Apply sin and cos functions element-wise to the scaled_time tensor.

$$sin\_emb=sin(scaled\_time)$$

$$cos\_emb=cos(scaled\_time)$$

6. <b>Concatenate</b>: Combine the sine and cosine tensors along the last dimension.

The final output is a tensor of shape [$B$, $D$]. Each row in this tensor is the high-dimensional vector embedding for the corresponding scalar timestep $t$ that was in the input batch.


### Training Process ###

For each image $x_0$​ in a batch, the model learns by performing the following steps:

1. Sample Timestep & Noise:

   - Pick a random timestep $t$ from the diffusion schedule (e.g., $t\in[1,T]$).
   - Generate a $GT$ noise sample $\epsilon$ from a standard normal distribution, with the same size as the original image.

2. Create Noisy Image (Forward Process):

   - Create the noisy image $x_t$​ by directly "diffusing" the original image $x_0$​ to timestep $t$. $\alpha_t$ is the variance schedule, making $\sqrt{\bar \alpha_t}$ the signal rate (how much of the original we want to keep).
    - They are mixed using $x_t = \sqrt{\bar \alpha_t} x_0 + \sqrt{1 - \bar \alpha_t} \epsilon$. 

3. Predict Noise (Reverse Process):

   - Feed the noisy image $x_t$​, the timestep $t$ (using its sinusoidal embedding), and the conditioning (class label $y$) into the U-Net model.
   - The model's goal is to predict the original noise that was added ($\epsilon_\theta$).

4. Calculate Loss:

   - Compute the difference between the actual noise $\epsilon$ (from step 1) and the predicted noise $\epsilon_\theta$​ (from step 3).
   - The loss is the Mean Squared Error (MSE): $L=MSE(\epsilon,\epsilon_\theta​)$.

The model's weights are then updated, teaching the U-Net to predict the noise added at any given timestep.

### Generation Process ###

We generate $x_T$ images by sampling from a standard gaussian distribution. and start iterating backwards from $T$ to $0$
1. Feed the noisy image $x_t$ to the model to obtain predicted noise $\epsilon_\theta$.
2. Calculate the mean of the posterior distribution. 
   - $\beta_t$ are a set of $T$ small constrants typically increasing linearly.
   - Alpha schedule: $\alpha_t​=1−\beta_t$. 
   - $\bar \alpha_t$ is the cumulative product of $\alpha$ up to $t$.

$$\mu_\theta(x_t,t) = \frac{1}{\sqrt \alpha_t}(x_t-\frac{\beta_t}{\sqrt{1-\bar \alpha_t}}\epsilon_\theta(x_t,t,y))$$

3. If $t>0$ add small amount of noise $z \sim \mathcal{N}(0,I)$ for stochasticity 
   - $x_{t-1} = \mu_\theta(x_t,t) + \sigma_t z$
   - Variance of the true posterior distribution: $\sigma_t = \sqrt{\frac{1-\bar \alpha_{t-1}}{1- \bar \alpha_t}*\beta_t}$
   - The variance of the posterior distribution is derived from the bayes theorem. Because our posterior distributions are gaussians, the following equation can be used to derive the above formula.

$$ q(x_{t-1}|x_t,x_0) = \frac{q(x_t|x_{t-1},x_0)*q(x_{t-1}|x_0)}{q(x_t|x_0)} $$


4. Finally we rescale the output $x_{t-1}$ from $[-1,1]$ to $[0,1]$.