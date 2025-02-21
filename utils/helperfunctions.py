from cv2 import cvtColor, COLOR_RGB2GRAY, calcHist
from numpy import zeros
from torch import no_grad
from torch.nn.functional import cross_entropy
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# from tqdm import tqdm


def compute_average_histogram(images):
    '''
    Compute average histogram of a given dataset
    '''
    # Initialize an empty array to accumulate histograms
    sum_hist = zeros((256,))
 
    # Compute histogram for each image and accumulate
    for image in images:
        if image.shape[-1] == 3:
            image = cvtColor(image, COLOR_RGB2GRAY)
        else:
            image = image.numpy()
        hist = calcHist([image], [0], None, [256], [0, 256])
        sum_hist += hist.flatten()
    # Compute the average histogram
    avg_hist = sum_hist / len(images)
    return avg_hist


def evaluate(model, test_loader, device):
    total = 0
    loss = 0
    all_preds = []
    all_targets = []   
    print("Evaluating")
    model.eval()
    with no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss += cross_entropy(out, target, reduction="sum")
            pred = out.max(1)[1]
            total += len(target)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
        
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    # print(f"DEBUG: SKLEARN Accuracy: {accuracy_score}")
    # print(f"DEBUG: Calculated Accuracy: {(correct / total).item()}")
            
    return accuracy, (loss / total).item(), precision, recall, f1
