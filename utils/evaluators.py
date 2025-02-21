from numpy import log, var, sum, size, log2
from cv2 import cvtColor, COLOR_RGB2GRAY

def entropy(data):
    '''
    Compute grayscale entropy of data
    '''
    print("EVALUATOR: Entropy")
    return [-sum(x * log(x + 1e-7)) for x in data]


def combine_patch_entropies(data):
    '''
    Sum up all entropies from patch entropy vectors.
    '''
    entropies = []
    for vector in data:
        entropies.append(sum(vector))
    return entropies


def average_rgb_entropy(data):
    '''
    Compute average rgb entropy of data
    '''
    print("EVALUATOR: Average RGB Entropy")
    return [(-sum(x[0] * log(x[0] + 1e-7)) + 
            -sum(x[1] * log(x[1] + 1e-7)) + 
            -sum(x[2] * log(x[2] + 1e-7))) / 3 for x in data]


def image_variance(data):
    '''
    Compute the variance of pixel intensities
    '''
    print("EVALUATOR: Image Variance")
    variances = []
    for image in data.data:
        if image.shape[-1] == 3:
            image = cvtColor(image, COLOR_RGB2GRAY)
        else:
            image = image.numpy()
        variances.append(var(image))
    return variances