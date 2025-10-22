# simple-diffusion
Conditional diffusion models to generate MNIST characters.

We are using a U-Net with Sinusoidal Position Embeddings defining the timestep in the diffusion process.

### The Sinusoidal Position Embedding: ###

Maps the scalar t to a high-dimensional vector, giving the network richer information. This ensures continuity and a consistent meaning across different numbers of total timesteps. This helps the network learn the smooth transition from one noise level to the next.

The steps to create the sinusoidal position embedding are:

1. We determine the full embedding dimension D, and half embedding dimension (D/2), as we will concat sin and cos.

2. $$\beta = \frac{log(10000)}{D/2−1​}$$

3. Calculate Frequencies: Create a range of indices i from 0 to D/2−1. The inverse frequencies are calculated by applying the exponential function to these indices, scaled by $-β$:

$${inv\_freq}_i ​= e^{−i · β}$$

This results in a vector of shape $[D/2]$.

1. The time value t is now multiplied by each of the inverse frequencies calculated in Step 2. This creates the "scaled time" tensor.

Broadcast: The time tensor (shape [B]) is expanded to [B, 1]. The inv_freq vector (shape [D/2]) is expanded to [1, D/2].

Multiplication: An element-wise multiplication (an outer product) is performed, creating a tensor of shape [B, D/2]. Each time step in the batch is now scaled by all the pre-defined frequencies.

$${scaled\_time}_{b,i}​ = t_b​ ⋅ {inv\_freq}_i​$$

Code Line: embeddings = time[:, None] * embeddings[None, :]

5. he core of the sinusoidal encoding involves applying sine and cosine functions to the scaled time values.

Sine and Cosine: Apply the sin(⋅) function to the first half of the dimensions and the cos(⋅) function to the second half (or simply apply both to all dimensions, as done here).

$${sin\_emb}=sin({scaled\_time})$$

$${cos\_emb}=cos({scaled\_time})$$

- Concatenate the sine and cosine results along the last dimension.

Code Line: embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

Output: A tensor of shape [BatchSize, D]

### Training Process ###

1. Pick a random timestep $t$.
2. Generate pure noise $\epsilon$.
3. Use a formula to create a noisy image $x_t$​ for that timestep by mixing the original image $x_0$​ and the noise $\epsilon$.
4. Feed the noisy image $x_t$​, the timestep $t$, and the label $y$ to our U-Net.
5. The U-Net will predict some noise.
6. The loss is how different the U-Net's predicted noise is from the actual noise $\epsilon$ we added in step 2. We use Mean Squared Error (MSE) for this.