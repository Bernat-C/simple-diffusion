# simple-diffusion
Conditional diffusion models to generate MNIST characters.

The training process is as follows for each image in a batch:

- Pick a random timestep t.
- Generate pure noise ϵ.
- Use a formula to create a noisy image xt​ for that timestep by mixing the original image x0​ and the noise ϵ.
- Feed the noisy image xt​, the timestep t, and the label y to our U-Net.
- The U-Net will predict some noise.
- The loss is how different the U-Net's predicted noise is from the actual noise ϵ we added in step 2. We use Mean Squared Error for this.