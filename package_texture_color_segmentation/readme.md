# Texture Color Segmenation

A Python package for image segmentation that combines texture features with color edges to create precise object boundaries.

## Installation

To install the package, use the following command:

```bash
pip install texture-color-segmentation
```

For the example below, the Distance Regularized Level Set Evolution (DRLSE) segmentation algorithm is used. This algorithm is introduced in the paper "Distance Regularized Level Set Evolution and Its Application to Image Segmentation" by C. Li, C. Xu, C. Gui and M. D. Fox, in IEEE Transactions on Image Processing, vol. 19, no. 12, pp. 3243-3254, Dec. 2010, doi: 10.1109/TIP.2010.2069690.

An implementation of the DRLSE algorithm is included in this package. This implementation is based on the work of [Ramesh Pramuditha Rathnayake](https://github.com/Ramesh-X/Level-Set), with slight modifications to suit our specific use case. Good to know, the parameters chosen for the DRLSE-algoritm can be very sensitive for varying use cases. 

## Basic example

Basic usage example.

```python
from texture_color_segmentation.feature_extraction import feature_extractor
from texture_color_segmentation.distance_regularized_level_set_evolution.find_lsf import find_lsf
import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import cv2

# load image
image = cv2.imread( "path/to/image.jpg")

# Retrieve combined texture boundaries with color edges
extractor = feature_extractor(image, window_size = 15)
g_combined = extractor.get_combined_edges()

# Initialize contour
c0 = 2
initial_lsf = c0 * np.ones_like(g_combined)
initial_lsf[10:-10, 10:-10] = -c0

# Evolulion contour with DRLSE
phi = find_lsf(
        g = g_combined,
        initial_lsf = initial_lsf,
        timestep = 5,
        iter_inner = 20,
        iter_outer = 30,
        lmda = 5,
        alfa = 2,
        epsilon = 1.5)

# Find contours from zero-level set
contours = measure.find_contours(phi, 0)

# show results
plt.imshow(image)
for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
plt.show()
```

## feature_extractor Class

The `feature_extractor` class is designed for extracting and processing texture and color features from an image to facilitate image segmentation.

### Initialization

**Constructor**: `__init__(self, image, window_size, feature_configurations=[(0, [0, np.pi/4, np.pi/2, 3*np.pi/4], [1]), (1, [0, np.pi/4, np.pi/2, 3*np.pi/4], [1])], feature_methods=["contrast", "cluster prominence", "cluster shade"], wavelet_type="haar", comatrix_level=32)`

**Parameters**:
- `image` (numpy.ndarray): The input image to be processed.
- `window_size` (int): The size of the window used for texture feature extraction.
- `feature_configurations` (list of triplets): Configurations for the texture features. Each triplet contains:
  - `level` (int): The wavelet decomposition level.
  - `angles` (list of floats): Angles (in radians) for the gray-level co-occurrence matrix (GLCM).
  - `distances` (list of ints): Distances for the GLCM.
- `feature_methods` (list of strings): The methods for texture feature extraction (possible values: "contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation", "cluster prominence", "cluster shade").
- `wavelet_type` (str): The type of wavelet used for decomposition (default is "haar").
- `comatrix_level` (int): The number of gray levels for the co-occurrence matrix (default is 32).

### Methods

#### get_texture_features

**Description**: Extracts texture features from the image using wavelet and GLCM-based methods.

**Returns**:
- `texture_features` (numpy.ndarray): The extracted texture features.

#### get_combined_texture_features

**Description**: Processes and combines the extracted texture features by applying Gaussian filters and normalization.

**Returns**:
- `combined_texture_features` (numpy.ndarray): The combined and processed texture features.

#### get_texture_edges

**Description**: Generates texture edges from the combined texture features using a specified edge indicator and structuring element.

**Parameters**:
- `edge_indicator` (function): The function used to indicate edges (default is `scalar_differencing`).
- `sigma_structuring_element` (float): The sigma value for the Gaussian structuring element (default is 1.5).

**Returns**:
- `texture_edges` (numpy.ndarray): The detected texture edges.

#### get_color_edges

**Description**: Detects edges based on color information using a bilateral filter and a specified edge indicator.

**Parameters**:
- `edge_indicator` (function): The function used to indicate edges (default is `scalar_differencing`).
- `blur_diameter` (int): Diameter of each pixel neighborhood used during the bilateral filter (default is 9).
- `blur_sigma` (float): Filter sigma in the color space (default is 75).

**Returns**:
- `color_edges` (numpy.ndarray): The detected color edges.

#### get_combined_edges

**Description**: Combines texture and color edges based on a homogeneity threshold to produce final segmentation edges.

**Parameters**:
- `edge_indicator` (function): The function used to indicate edges (default is `scalar_differencing`).
- `homogeneity_threshold` (float): The threshold for determining homogeneous areas (default is 0.9).
- `sigma_structuring_element` (float): The sigma value for the Gaussian structuring element (default is 1.5).
- `blur_diameter` (int): Diameter of each pixel neighborhood used during the bilateral filter (default is 9).
- `blur_sigma` (float): Filter sigma in the color space (default is 75).

**Returns**:
- `combined_edges` (numpy.ndarray): The combined texture and color edges.