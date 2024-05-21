import numpy as np
from scipy.ndimage import gaussian_filter

def cluster_shade(comat, r, c, Mx, My):
    """Calculates the cluster shade feature from the co-occurrence matrix."""
    term1 = np.power((r - Mx + c - My), 3)
    term2 = comat.flatten()

    return np.sum(term1 * term2)

def cluster_prominence(comat, r, c, Mx, My):
    """Calculates the cluster prominence feature from the co-occurrence matrix."""
    term1 = np.power((r - Mx + c - My), 4)
    term2 = comat.flatten()

    return np.sum(term1 * term2)

def normalize(image):
    """Normalizes the image intensity values to the range [0, 1]."""

    image_min = np.min(image)
    image_max = np.max(image)
    normalized_image = (image - image_min) / (1 if image_max == image_min else abs(image_max - image_min))

    return normalized_image

def log_normalize(image):
    """Logarithmically normalizes the image intensity values to the range [0, 1]."""

    def log_modulus(x):
        return np.sign(x) * np.log(np.abs(x) + 1)
    vlog_modulus = np.vectorize(log_modulus)

    log_modulus_image = vlog_modulus(image)
    
    return normalize(log_modulus_image)

def invert(X):
    X_min = np.min(X)
    X_max = np.max(X)

    return (((X - X_min) - (X_max - X_min)) * -1) + X_min

def gaussian_disk(radius, sigma):
    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(x, y)
    distances = np.sqrt(xx**2 + yy**2)
    disk = distances <= radius
    gaussian = gaussian_filter(disk.astype(float), sigma=sigma)
    gaussian /= np.sum(gaussian)
    return gaussian

def scalar_differencing(input):
    [Iy, Ix] = np.gradient(input)
    f = np.square(Ix) + np.square(Iy)
    return 1 / (1 + f)