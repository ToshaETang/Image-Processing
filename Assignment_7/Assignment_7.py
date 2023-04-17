# python Assignment_7.py

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Create the input image g(x,y) with a gray value of 100
g = Image.new('L', (300, 300), color=100)
g.show()

# Display the histogram
plt.hist(np.array(g).ravel(), bins=256, range=(0, 255))
plt.title('Histogram of Original Image')
plt.show()

# Generate the Gaussian noise n(x,y) with mu=0 and sigma^2=25
mu, sigma = 0, 5
noise = np.random.normal(mu, sigma, size=(300, 300))

# Add the noise to the input image to create the noisy image f(x,y)
f = np.array(g) + noise

# Display the noisy image f(x,y)
plt.imshow(f, cmap='gray')
plt.title('Noisy Image')
plt.show()

# Display the histogram
plt.hist(f.ravel(), bins=256, range=(0, 255))
plt.title('Histogram of Noisy Image')
plt.show()

