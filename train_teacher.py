import sys, os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir) 

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import time
import numpy as np
from transformers import ViTForImageClassification

# parsers
parser = argparse.ArgumentParser(description='Train teacher model')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'mnist', 'fashionmnist'], type=str, help='Dataset')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--opt', default="adamw")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--bs', default='16')
parser.add_argument('--size', default="224")
parser.add_argument('--n_epochs', type=int, default='12')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--output-dir', default='teachers/', help='path where to save teacher weights')
parser.add_argument('--accumulation-steps', default='8', type=int, help='The accumulation_steps parameter sets the number of mini-batches over which you accumulate gradients before performing a weight update.')
parser.add_argument("--device", help="The device to use", choices=['cuda','cpu'], default='cuda')
parser.add_argument('--seed', type=int, default=None)

args = parser.parse_args()

best_acc = 0  # best test accuracy

def finetune(epoch, use_amp, net, trainloader, optimizer, criterion, device):
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    output_logits = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.amp.autocast(enabled=use_amp, device_type=args.device):
            outputs = net(inputs)
            logits = outputs.logits  # Get logits from the last layer
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        train_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        output_logits.extend(logits.cpu().detach().float().numpy())  
    return train_loss/(batch_idx+1), output_logits


def finetune_test(epoch, net, testloader, criterion, optimizer, device, save_as=None):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs).logits
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        os.makedirs(f'{args.output_dir}/{args.dataset}/', exist_ok=True)
        if save_as is None:
            save_as = os.path.join(f'{args.output_dir}/{args.dataset}/', "{}.pth".format('deit-tiny-patch16-224') )
        print("Directory: ", save_as)
        torch.save(net.state_dict(), save_as)
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    os.makedirs(f"log/{args.dataset}", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/{args.dataset}/log_deit_tiny_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc


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

    bs = int(args.bs)
    imsize = int(args.size)
    use_amp = not args.noamp
    device = args.device
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')

    if args.dataset == 'cifar10':
        num_classes = 10
        # CIFAR
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)


    elif args.dataset == 'mnist':
        num_classes = 10
        # MNIST
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)


        transform_test = transforms.Compose([ 
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])            
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    
    elif args.dataset == 'fashionmnist':
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)


        transform_test = transforms.Compose([ 
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])            
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=4)
    print("Dataset Loaded")

    print('==> Building model..')    
    net = ViTForImageClassification.from_pretrained('facebook/deit-tiny-patch16-224') 
    net.classifier = nn.Linear(net.config.hidden_size, num_classes)

    print("Selected Device: ", device)
    net.to(device)

    # Loss is CE
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if args.opt == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.opt == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr)
    elif args.opt =='adamw':  
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)

    print("Optimizer:", optimizer)

    list_loss = []
    list_acc = []

    net.to(args.device)
    
    logits = None
    for epoch in range(start_epoch, args.n_epochs):
        print(f"Epoch {epoch}")

        _, logits = finetune(epoch, use_amp, net, trainloader, optimizer, criterion, device)
        val_loss, acc = finetune_test(epoch, net, testloader, criterion, optimizer, device) 
        
        list_loss.append(val_loss)
        list_acc.append(acc)    
    
    np.save(f'./logits/{args.dataset}.npy', np.array(logits))


if __name__ == "__main__":
    main()