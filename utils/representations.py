from cv2 import cvtColor, COLOR_RGB2GRAY, calcHist
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate, size, log2
from scipy.special import softmax

def average_adjusted_histograms(dataset):
    from numpy import zeros
    print("REPRESENTATIONS: Average Adjusted Histograms")
    #Calculate histogram of each dataset sample
    histograms = []
    for image in dataset.data:
        if image.shape[-1] == 3:
            image = cvtColor(image, COLOR_RGB2GRAY)
        else:
            image = image.numpy()
        histograms.append(calcHist([image], [0], None, [256], [0, 256]).flatten())

    # Calulcate average histogram
    sum_hist = zeros((256,))
    for hist in histograms:
        sum_hist += hist.flatten()
    avg_hist = sum_hist / len(histograms)

    representations = []
    for hist in histograms:        
        # Normalize the histograms using MinMaxScaler
        scaler = MinMaxScaler()
        norm_hist = scaler.fit_transform(hist.reshape(-1, 1)).flatten()
        norm_avg_hist = scaler.transform(avg_hist.reshape(-1, 1)).flatten()
        # Compute the absolute difference between normalized histograms
        representations.append(abs(norm_hist - norm_avg_hist))
    return representations


def histograms_rgb(dataset):
    '''
    Compute rgb histograms of dataset. 
    Returns an array of dimensions 1 x h*w*c of histogram bins for each
    color channel for each image.
    '''
    print("REPRESENTATIONS: RGB Histograms")
    histograms = []
    for image in dataset.data:
        r = calcHist(images=[image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
        g = calcHist(images=[image], channels=[1], mask=None, histSize=[256], ranges=[0, 256])
        b = calcHist(images=[image], channels=[2], mask=None, histSize=[256], ranges=[0, 256])
        hist = concatenate((r, g, b), axis = 0).squeeze()
        hist /= hist.sum()
        histograms.append(hist)
    return histograms


def histograms_grayscale(dataset):
    '''
    Compute grayscale histograms of dataset
    '''
    print("REPRESENTATIONS: Grayscale Histograms")
    histograms = []
    for image in dataset.data:
        if image.shape[-1] == 3:
            gray_image = cvtColor(image, COLOR_RGB2GRAY)
        else:
            gray_image = image.numpy()    
        hist = calcHist([gray_image], [0], None, [256], [0, 256])
        hist /= hist.sum() #Normalise by the total number of pixels to convert to probability distribution as described in paper. (32x32=1024 CIFAR 28x28=784 MNIST)

        histograms.append(hist.squeeze())

    return histograms


def patch_entropy_vectors(data):
    '''
    Compute the entropy of image regions and return the values as a
    vector.
    '''
    N = 8
    representations = []
    print("REPRESENTATIONS: Patch Entropy Vectors")
    for colorIm in data.data:
        image_patches_entropy_vector = []
        if colorIm.shape[-1] == 3:
            greyIm = cvtColor(colorIm, COLOR_RGB2GRAY)
        else:
            greyIm = colorIm.numpy()
        tiles = [greyIm[x:x+N,y:y+N] for x in range(0,greyIm.shape[0],N) for y in range(0,greyIm.shape[1],N)]
        for tile in tiles:
            signal = tile.flatten()
            lensig=signal.size
            symset=list(set(signal))
            propab=[size(signal[signal==i]) / lensig for i in symset]
            ent=sum([p*log2(1.0/p) for p in propab])
            image_patches_entropy_vector.append(ent)
        representations.append(image_patches_entropy_vector)
    return representations


def compressed_feature_vectors(dataset):
    '''
    Train an autoencoder and compute the compressed feature vector of the dataset samples.
    '''
    from utils.autoencoder import Autoencoder, train_autoencoder
    from torch.utils.data import DataLoader
    from torch import no_grad, cuda
    from tqdm import tqdm

    print("REPRESENTATIONS: Compressed Feature Vectors")
    autoencoder = Autoencoder()    
    trainloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    device = 'cuda' if cuda.is_available() else 'cpu'
    train_autoencoder(autoencoder=autoencoder, trainloader=trainloader, device=device)

    compressed_vectors = []

    with no_grad():
        for data in tqdm(trainloader, total=len(trainloader), desc=f"Calculating compressed feature vectors"):
            inputs, _ = data
            inputs = inputs.to(device)
            encoded = autoencoder.encoder(inputs)            
            flattened_encoded = encoded.view(encoded.size(0), -1)
            compressed_vectors.extend(flattened_encoded.to('cpu').numpy())
    return compressed_vectors



def logits_vectors(dataset):
    from numpy import load
    from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
    print("REPRESENTATIONS: Logits Vectors")
    if isinstance(dataset, CIFAR10): 
        representations = load('./logits/logits_cifar10.npy', allow_pickle=True)        
    elif isinstance(dataset, MNIST):
        representations = load('./logits/logits_mnist.npy', allow_pickle=True)
    elif isinstance(dataset, FashionMNIST):
        representations = load('./logits/logits_fashionmnist.npy', allow_pickle=True)
    representations = [softmax(x) for x in representations] #convert to probabilities as described in paper
    return representations
