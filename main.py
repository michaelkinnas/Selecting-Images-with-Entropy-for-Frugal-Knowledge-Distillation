import sys, os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir) 

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
from utils.distillers import ResponseDistiller
from students import vgg_based_students
from transformers import ViTForImageClassification
from utils.reporting import report_to_std_out, report_to_file
from utils.helperfunctions import evaluate

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=['cifar10', 'mnist', 'fashionmnist'], required=True)

parser.add_argument("--method", type=str, 
                    choices=['top_n', 'top_n_per_category', 'kmeans', 'kmeans_per_category', 'manifold_learning', 'tn', 'tnpc', 'km', 'kmpc', 'ml'], 
                    help="Which overall sample selection method to use. You can use full name or abbreviated version", required=True)

parser.add_argument("--n-clusters", type=int, default=None, required=False, 
                    help="Optional. Only relevant if clustering method that receives number of clusters as argument is selected. If left as `None` an automatic algorithm for determining optimal number of clusters will be used, such as silhouette score. Warning: slow.")

parser.add_argument("--representations", type=str, 
                    choices=['histograms_grayscale', 'average_adjusted_histograms', 'histograms_rgb', 'compressed_feature_vectors', 'logits_vectors','patch_entropy_vectors', 'hg','aah','hrgb','cfv','lv', 'pev'], 
                    help="Used for the type of image representation method with normal and clustering methods.", required=False, default=None)

parser.add_argument("--evaluation", type=str, choices=["entropy", "variance"], default="entropy", required=False, 
                    help="Which evaluation method to use for the image scores. Default is entropy for which a representation is required.")

parser.add_argument("--selection", type=str, choices=["highest_score", "lowest_score", "random", "hs", "ls", "rng"],
                    help="The sample selection criterion based upon the calculated score (entropy or other)", required=False, default=None)

parser.add_argument("--n-samples", type=int, help="The number of samples to use", required=True)

parser.add_argument('--size', default="224", type=int) # TODO: Fix Compatibility with 32

parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--total-epochs", type=int, default=100)
parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
parser.add_argument("--use-validation-step", help='If set, a validation step will take place after each training epoch step.', action='store_true', default=False)
parser.add_argument("--evaluate", help='If set, an evaluation will take place after the end of training.', action='store_true', default=False)
parser.add_argument("--device", help="The device to use", choices=['cuda','cpu'], default='cuda')
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--report", type=str, default='stdout', choices=['stdout','file'])
parser.add_argument("--root-password", type=str, help="The root password to command the computer to shutdown. If not set the computer will remain on.", required=False, default=None)
args = parser.parse_args()

def main():
    # Define Execution Seed
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Define your Device
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if args.seed is not None:
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True

    # Load Data
    print('==> Preparing data..')
    '''
    Dataset logic
    '''
    if args.dataset == 'cifar10':
        # CIFAR
        transform_train = transforms.Compose([
            transforms.Resize(args.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        
        if args.use_validation_step or args.evaluate:
            transform_test = transforms.Compose([
                transforms.Resize(args.size),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)


    elif args.dataset == 'mnist':
        # MNIST
        transform_train = transforms.Compose([
            transforms.Resize(args.size),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)

        if args.use_validation_step or args.evaluate:
            transform_test = transforms.Compose([ 
                transforms.Resize(args.size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])            
            testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    
    elif args.dataset == 'fashionmnist':
        transform_train = transforms.Compose([
            transforms.Resize(args.size),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)

        if args.use_validation_step or args.evaluate:
            transform_test = transforms.Compose([ 
                transforms.Resize(args.size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])            
            testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    

    # input("RESET POWER METER AND PRESS ENTER")
    process_start_time = time.time()


    #METHOD
    if args.selection in ["random", "rng"]:            
        import random
        if args.seed is not None:
            random.seed(args.seed)
        selected_indices = random.choices(range(0, len(trainset)), k=args.n_samples)

    elif args.method in ["manifold_learning", "ml"]:
        from utils.methods import manifold_learning

        selected_indices = manifold_learning(images=trainset.data,
                                            n_samples=args.n_samples,
                                            tsne_n_components=2,
                                            kmeans_clusters=100,
                                            seed=args.seed) #values from old code
    
    else:
        # REPRESENTATIONS
        if args.representations is None and args.evaluation != "variance":
            raise ValueError(f"Must choose representation method when using {args.evaluation} evaluation method.")
        
        if args.representations in ['histograms_grayscale','hg']:
            from utils.representations import histograms_grayscale
            representations = histograms_grayscale
        elif args.representations in ['histograms_rgb','hrgb']:
            from utils.representations import histograms_rgb
            if args.dataset != "cifar10":
                raise ValueError(f"This representation method is only available for the CIFAR-10 dataset.")
            representations = histograms_rgb
        elif args.representations in ['compressed_feature_vectors','cfv']:
            from utils.representations import compressed_feature_vectors
            representations = compressed_feature_vectors
        elif args.representations in ['logits_vectors','lv']:
            from utils.representations import logits_vectors
            representations = logits_vectors
        elif args.representations in ['average_adjusted_histograms','aah']:
            from utils.representations import average_adjusted_histograms
            representations = average_adjusted_histograms
        elif args.representations in ['patch_entropy_vectors','pev']:
            from utils.representations import patch_entropy_vectors
            representations = patch_entropy_vectors
        else: representations = None

        #EVALUATION
        if args.evaluation == "entropy":
            if args.representations is None:
                raise ValueError(f"You must use an image representation method when using 'Entropy' evaluation.")
                        
            if args.representations in ['patch_entropy_vectors', 'pev']:
                from utils.evaluators import combine_patch_entropies
                evaluator = combine_patch_entropies
            else:
                from utils.evaluators import entropy
                evaluator = entropy
        elif args.evaluation == "variance":
            from utils.evaluators import image_variance
            evaluator = image_variance
        elif args.evaluation == "patches_entropies_sum":
            if args.representations not in ['patch_entropy_vectors','pev']:
                raise ValueError("Patches entropy sum only works with patch entropy vectors.")

        if args.method in ["top_n", "tn"]:
            from utils.methods import top_n

            if args.selection in ['highest_score', 'hs']:
                from utils.selectors import highest_score
                selector = highest_score
            else:
                raise NotImplementedError("This selection has not been implemented yet for this method.")

            selected_indices = top_n(data=trainset,
                                    labels=trainset.targets, 
                                    n_samples=args.n_samples,
                                    representations_fn=representations,
                                    evaluator_fn=evaluator,
                                    selector_fn=selector
                                    )
        
        elif args.method in ["top_n_per_category", "tnpc"]:
            from utils.methods import top_n_per_category

            if args.selection is None:
                raise ValueError("You must use a selection method.")

            if args.selection in ['highest_score', 'hs']:
                from utils.selectors import highest_score_per_category
                selector = highest_score_per_category
            elif args.selection in ['lowest_score', 'ls']:
                from utils.selectors import lowest_score_per_category
                selector = lowest_score_per_category

            selected_indices = top_n_per_category(data=trainset,
                                                labels=trainset.targets, 
                                                n_samples=args.n_samples,
                                                representations_fn=representations,
                                                evaluator_fn=evaluator,
                                                selector_fn=selector
                                                )
        
        elif args.method in ["kmeans", "km"]:
            from utils.methods import kmeans_clustering

            if args.selection is None:
                raise ValueError("You must use a selection method.")

            if args.selection in ['highest_score', 'hs']:
                from utils.selectors import highest_score_per_cluster
                selector = highest_score_per_cluster
            elif args.selection in ['lowest_score', 'ls']:
                from utils.selectors import lowest_score_per_cluster
                selector = lowest_score_per_cluster


            selected_indices, n_clusters = kmeans_clustering(data=trainset,
                                                            n_samples=args.n_samples,                                                           
                                                            representations_fn=representations,
                                                            evaluator_fn=evaluator,
                                                            selector_fn=selector,
                                                            scaling=args.scaling,
                                                            lsa_components=args.n_dimensions,
                                                            n_clusters=args.n_clusters,
                                                            seed=args.seed)
        
        elif args.method in ["kmeans_per_category", "kmpc"]:
            from utils.methods import kmeans_clustering_per_category

            if args.selection in ['highest_score', 'hs']:
                from utils.selectors import highest_score_per_cluster
                selector = highest_score_per_cluster
            elif args.selection in ['lowest_score', 'ls']:
                from utils.selectors import lowest_score_per_cluster
                selector = lowest_score_per_cluster
     

            selected_indices, n_clusters = kmeans_clustering_per_category(data=trainset,
                                                                        labels=trainset.targets,
                                                                        n_samples=args.n_samples,                                                                     
                                                                        representations_fn=representations,
                                                                        evaluator_fn=evaluator,
                                                                        selector_fn=selector,
                                                                        scaling=args.scaling,
                                                                        lsa_components=args.n_dimensions,
                                                                        n_clusters=args.n_clusters,
                                                                        seed=args.seed)


    trainset_subset = torch.utils.data.Subset(trainset, selected_indices)
    trainloader = torch.utils.data.DataLoader(trainset_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    if args.use_validation_step or args.evaluate:
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print("Number of images before selection:", len(trainset))
    print("Number of images selected:", len(trainset_subset))

    #Load teacher
    
    teacher = ViTForImageClassification.from_pretrained('facebook/deit-tiny-patch16-224')
    teacher.classifier = nn.Linear(teacher.config.hidden_size, 10)
    loaded = torch.load(f'./teachers/{args.dataset}/deit-tiny-patch16-224.pth', map_location="cpu", weights_only=True)
    if isinstance(loaded, nn.Module):
        teacher = loaded
    else:
        teacher.load_state_dict(loaded)
    teacher = teacher.to(args.device)

    #Load student
    student = vgg_based_students.vgg19_bn(num_classes=10)

    optimizer = torch.optim.SGD(params=student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = F.cross_entropy


    distiller = ResponseDistiller(teacher=teacher,
                                  student=student,
                                  loss_fn=criterion,
                                  optimizer=optimizer,
                                  train_dataloader=trainloader,
                                  validation_dataloader=testloader,
                                  epochs=args.total_epochs,
                                    alpha=0.5,
                                    temperature=2,
                                    device=args.device,
                                    seed=args.seed,
                                  )
    
    distiller.distill(record_train_progress=False,
                      record_validation_progress=False,
                      verbose=True,
                      use_tqdm=False,
                      batch_reporting_step=max(1, int(len(trainloader) * 0.2))) #report only validation after each epoch
   
    process_end_time = time.time()

    if args.evaluate:
        acc, val_loss, precision, recall, f1 = evaluate(student, testloader, device=args.device)
    else:
        acc, val_loss, precision, recall, f1 = None, None, None, None, None
   
    if args.method not in ['kmeans', 'kmeans_per_category', 'km', 'kmpc']:
        n_clusters = None

    if args.report == 'stdout':        
        report_to_std_out(args, n_clusters, trainset_subset, process_start_time, process_end_time, 
                           acc=acc, precision=precision, recall=recall, f1=f1, val_loss=val_loss)

    elif args.report == 'file':
        report_to_file(f"./results/report_{args.dataset}.txt", args, n_clusters, trainset_subset, process_start_time, process_end_time, 
                           acc=acc, precision=precision, recall=recall, f1=f1, val_loss=val_loss)
    
    if args.root_password:
        os.system(f"echo {args.root_password} | sudo -S poweroff")

if __name__ == "__main__":
    main()
