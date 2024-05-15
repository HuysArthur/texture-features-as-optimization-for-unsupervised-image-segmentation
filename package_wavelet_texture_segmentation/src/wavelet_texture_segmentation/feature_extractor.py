import numpy as np
import pywt
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import generic_filter

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

def wavelet_coefficient_features(block, feature_configurations, feature_methods, wavelet_type, comatrix_level):
    block = block - np.mean(block, dtype=np.uint8) # idea from Mr Meert Wannes, maybe do something with negative values

    levels = [configuration[0] for configuration in feature_configurations]

    A = block.copy()
    wavelet_levels_edges = []
    for i in range(max(levels)):
        A, edges = pywt.dwt2(A, wavelet_type)
        wavelet_levels_edges.append(edges)


    comatrices = np.array([], dtype=np.int64).reshape(comatrix_level,comatrix_level,0)
    for configuration in feature_configurations:
        level = configuration[0]
        angles = configuration[1]
        distances = configuration[2]

        if level == 0:
            subimage = (block/(256/comatrix_level)).astype(np.uint8)

            c = graycomatrix(subimage, distances, angles, levels=comatrix_level, normed=True)
            c = c.reshape((c.shape[0], c.shape[1], c.shape[2]*c.shape[3]))
            comatrices = np.concatenate((comatrices, c), axis=2)
        else:
            for angle in angles:
                wavelet_edges = wavelet_levels_edges[level-1]

                subimage
                if angle in [0]: # apply to horizontal edges DWT
                    subimage = wavelet_edges[0]
                elif angle in [np.pi/2]: # apply to vertical edges DWT
                    subimage = wavelet_edges[1]
                elif angle in [np.pi/4, 3*np.pi/4]: # apply to diagonal edges DWT
                    subimage = wavelet_edges[2]
                else:
                    raise Exception(f"{angle} is not a valid angle to apply on a wavelet decomposition")
                subimage = ((subimage+256)/(256*2/comatrix_level)).astype(np.uint8)

                c = graycomatrix(subimage, distances, [angle], levels=comatrix_level, normed=True)
                c = c.reshape((c.shape[0], c.shape[1], c.shape[2]*c.shape[3]))
                comatrices = np.concatenate((comatrices, c), axis=2)

    r, c = np.meshgrid(np.arange(comatrices.shape[0]), np.arange(comatrices.shape[1]))
    r = r.flatten() + 1
    c = c.flatten() + 1

    data = np.zeros((sum([len(configuration[1])*len(configuration[2]) for configuration in feature_configurations]),len(feature_methods)))

    for ci in range(comatrices.shape[2]):
        Mx = np.sum(r * comatrices[:,:,ci].flatten())
        My = np.sum(c * comatrices[:,:,ci].flatten())

        for i, feature in enumerate(feature_methods):
            if feature == "cluster":
                data[ci, i] = cluster_shade(comatrices[:,:,ci], r, c, Mx, My) + cluster_prominence(comatrices[:,:,ci], r, c, Mx, My)
            else:
                data[ci, i] = graycoprops(comatrices[:,:,ci].reshape(comatrix_level,comatrix_level,1,1), feature)[0,0]
    
    return data


def wavelet_texture_features_segmentation(gray_image, window_size, feature_configurations = [(0, [0, np.pi/4, np.pi/2, 3*np.pi/4], [1]),(1, [0, np.pi/4, np.pi/2, 3*np.pi/4], [1])], feature_methods = {"contrast", "cluster"}, wavelet_type="haar", comatrix_level=16):
    class fnc_class:
        def __init__(self, _array):
            # store the shape:
            self.shape = _array.shape
            self._array = _array
            # initialize the coordinates:
            self.coordinates = [0] * len(self.shape[:2])

            self.first_error = True
            
        def filter(self, buffer):
            try:
                size = int(len(buffer)**(1/2))
                block = buffer.reshape(size,size)

                self._array[self.coordinates[0], self.coordinates[1]] = wavelet_coefficient_features(block, feature_configurations, feature_methods, wavelet_type, comatrix_level)

                # calculate the next coordinates:
                axes = range(len(self.shape[:2]))
                for jj in reversed(axes):
                    if self.coordinates[jj] < self.shape[jj] - 1:
                        self.coordinates[jj] += 1
                        break
                    else:
                        self.coordinates[jj] = 0

                return 0
            except Exception as e:
                if self.first_error:
                    print(e)

                self.first_error = False
                raise e
        
    image_size = gray_image.shape
    WCFs = np.zeros((image_size[0], image_size[1], sum([len(configuration[1])*len(configuration[2]) for configuration in feature_configurations]), len(feature_methods)))

    fnc = fnc_class(WCFs)
    generic_filter(gray_image, fnc.filter, window_size)

    return WCFs