"""
This python code demonstrates an edge-based active contour model as an application of the
Distance Regularized Level Set Evolution (DRLSE) formulation in the following paper:

  C. Li, C. Xu, C. Gui, M. D. Fox, "Distance Regularized Level Set Evolution and Its Application to Image Segmentation",
     IEEE Trans. Image Processing, vol. 19 (12), pp. 3243-3254, 2010.

Author: Ramesh Pramuditha Rathnayake
E-mail: rsoft.ramesh@gmail.com

Released Under MIT License
"""
import numpy as np

from texture_color_segmentation.distance_regularized_level_set_evolution.drlse_algo import drlse_edge
from texture_color_segmentation.distance_regularized_level_set_evolution.potential_func import DOUBLE_WELL, SINGLE_WELL


def find_lsf(g: np.ndarray, initial_lsf: np.ndarray, timestep=1, iter_inner=10, iter_outer=30, lmda=5,
             alfa=-3, epsilon=1.5, potential_function=DOUBLE_WELL):
    """
    :param img: Input image as a grey scale uint8 array (0-255)
    :param initial_lsf: Array as same size as the img that contains the seed points for the LSF.
    :param timestep: Time Step
    :param iter_inner: How many iterations to run drlse before showing the output
    :param iter_outer: How many iterations to run the iter_inner
    :param lmda: coefficient of the weighted length term L(phi)
    :param alfa: coefficient of the weighted area term A(phi)
    :param epsilon: parameter that specifies the width of the DiracDelta function
    :param sigma: scale parameter in Gaussian kernal
    :param potential_function: The potential function to use in drlse algorithm. Should be SINGLE_WELL or DOUBLE_WELL
    """

    # parameters
    mu = 0.2 / timestep  # coefficient of the distance regularization term R(phi)

    # initialize LSF as binary step function
    phi = initial_lsf.copy()

    # start level set evolution
    for n in range(iter_outer):
        phi = drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_inner, potential_function)

    # refine the zero level contour by further level set evolution with alfa=0
    alfa = 0
    iter_refine = 10
    phi = drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_refine, potential_function)
    return phi
