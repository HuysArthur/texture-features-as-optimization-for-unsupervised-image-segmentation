import numpy as np
import pywt
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import generic_filter, gaussian_filter, grey_dilation
from skimage import metrics
import cv2

from wavelet_texture_segmentation import utils

def wavelet_texture_features(block, feature_configurations, feature_methods, wavelet_type, comatrix_level):
    block_mean = np.mean(block)
    if block_mean:
        block = block / block_mean
    block = block - np.min(block)

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
            subimage = (block/(256/block_mean))*comatrix_level
            subimage = np.clip(subimage, 0, comatrix_level-1).astype(np.uint8)

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
                
                subimage = ((subimage+((256/block_mean)*level))/(2*(256/block_mean)*level)) * comatrix_level
                subimage = np.clip(subimage, 0, comatrix_level-1).astype(np.uint8)

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
            if feature == "cluster prominence":
                data[ci, i] = utils.cluster_prominence(comatrices[:,:,ci], r, c, Mx, My)
            elif feature == "cluster shade":
                data[ci, i] = utils.cluster_shade(comatrices[:,:,ci], r, c, Mx, My)
            else:
                data[ci, i] = graycoprops(comatrices[:,:,ci].reshape(comatrix_level,comatrix_level,1,1), feature)[0,0]
    
    return data


def wavelet_texture_features_segmentation(gray_image, window_size, feature_configurations = [(0, [0, np.pi/4, np.pi/2, 3*np.pi/4], [1]),(1, [0, np.pi/4, np.pi/2, 3*np.pi/4], [1])], feature_methods = ["contrast", "cluster prominence", "cluster shade"], wavelet_type="haar", comatrix_level=16):
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

                self._array[self.coordinates[0], self.coordinates[1]] = wavelet_texture_features(block, feature_configurations, feature_methods, wavelet_type, comatrix_level)

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

    if window_size != 0:
        fnc = fnc_class(WCFs)
        generic_filter(gray_image, fnc.filter, window_size)

    return WCFs

class feature_extractor():
    def __init__(self, image, window_size, feature_configurations = [(0, [0, np.pi/4, np.pi/2, 3*np.pi/4], [1]),(1, [0, np.pi/4, np.pi/2, 3*np.pi/4], [1])], feature_methods = ["contrast", "cluster prominence", "cluster shade"], wavelet_type="haar", comatrix_level=16):
        self.image = image
        self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        self.window_size = window_size
        self.feature_configurations = feature_configurations
        self.feature_methods = feature_methods
        self.wavelet_type = wavelet_type
        self.comatrix_level = comatrix_level
    
        self.texture_features = None
        self.homogeneity_features = None
    
    def get_texture_features(self):
        if type(self.texture_features) == type(None):
            print("Extracting texture features, this can take a while")

            self.texture_features = wavelet_texture_features_segmentation(gray_image=self.gray_image, window_size=self.window_size, feature_configurations=self.feature_configurations, feature_methods=self.feature_methods, wavelet_type=self.wavelet_type, comatrix_level=self.comatrix_level)
        return self.texture_features
    
    def get_texture_edges(self, edge_indicator=utils.scalar_differencing, sigma_structuring_element=2):
        features = self.get_texture_features()

        processed_features = np.zeros_like(features)
        for i in range(features.shape[2]):
            for j in range(features.shape[3]):
                processed_features[:,:, i, j] = gaussian_filter(features[:,:,i,j], 1.5)

                if ("cluster prominence" in self.feature_methods and j == self.feature_methods.index("cluster prominence")) or ("cluster shade" in self.feature_methods and j == self.feature_methods.index("cluster shade")):
                    processed_features[:,:, i, j] = utils.log_normalize(processed_features[:,:,i, j])
                else:
                    processed_features[:,:, i, j] = utils.normalize(processed_features[:,:,i, j])

        std_deviations = np.std(processed_features, axis=(0, 1))
        i = std_deviations.flatten().argsort()[-1]
        highest_deviation = processed_features[:,:,i//features.shape[3], i%features.shape[3]]

        for i in range(features.shape[2]):
            for j in range(features.shape[3]):
                if metrics.mean_squared_error(highest_deviation, processed_features[:,:,i,j]) > metrics.mean_squared_error(highest_deviation, utils.invert(processed_features[:,:,i,j])):
                    processed_features[:,:, i, j] = utils.invert(processed_features[:,:,i, j])
        
        combined_features = np.mean(np.mean(processed_features, -1), -1)

        g_texture = edge_indicator(combined_features * 256)

        return grey_dilation(g_texture, structure=utils.gaussian_disk(int(self.window_size/2), sigma_structuring_element) * 256)
    
    
    def get_color_edges(self, edge_indicator=utils.scalar_differencing, blur_diameter=9, blur_sigma=75):
        edge_preserving_blur = cv2.bilateralFilter(self.image, blur_diameter, blur_sigma, blur_sigma)
        gray_smooth = cv2.cvtColor(edge_preserving_blur, cv2.COLOR_BGR2GRAY).astype(np.uint8)

        return edge_indicator(gray_smooth)
    
    def get_combined_edges(self, edge_indicator=utils.scalar_differencing, homogeneity_threhold=0.9, sigma_structuring_element=2, blur_diameter=9, blur_sigma=75):
        g_texture = self.get_texture_edges(edge_indicator=edge_indicator, sigma_structuring_element=sigma_structuring_element)
        g_color = self.get_color_edges(edge_indicator=edge_indicator, blur_diameter=blur_diameter, blur_sigma=blur_sigma)

        normalized_g_color = utils.normalize(g_color)
        normalized_g_texture = utils.normalize(g_texture)

        homogeneous_area = self.__homogenous_area(threshold=homogeneity_threhold, sigma_structuring_element=sigma_structuring_element)

        g_combined = np.zeros_like(self.gray_image).astype(np.float64)
        for i in range(self.gray_image.shape[0]):
            for j in range(self.gray_image.shape[1]):
                if homogeneous_area[i,j]:
                    g_combined[i,j] = normalized_g_color[i,j]
                else:
                    g_combined[i,j] = normalized_g_texture[i,j]
        
        return g_combined
    
    def __homogenous_area(self, threshold, sigma_structuring_element):
        homogeneity = np.zeros_like(self.gray_image)

        if "homogeneity" in self.feature_methods:
            features = self.get_texture_features()

            homogeneity_index = self.feature_methods.index("homogeneity")
            homogeneity = np.min(features[:,:,:,homogeneity_index], -1)
        else:
            if type(self.homogeneity_features) == type(None):
                print("Extracting homogeneity features, this can take a bit")
                self.homogeneity_features = wavelet_texture_features_segmentation(gray_image=self.gray_image, window_size=self.window_size, feature_configurations=self.feature_configurations, feature_methods=["homogeneity"], wavelet_type=self.wavelet_type, comatrix_level=self.comatrix_level)
            
            homogeneity = np.min(self.homogeneity_features[:,:,:,0], -1)
        
        homogeneity = grey_dilation(homogeneity, structure=utils.gaussian_disk(int(self.window_size/2), sigma_structuring_element))
        
        return homogeneity >= threshold