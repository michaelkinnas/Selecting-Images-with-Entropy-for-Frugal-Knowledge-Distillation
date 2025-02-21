# Selecting Images With Entropy For Frugal Knowledge Distillation

This repository contains the PyTorch implementation of the paper [Selecting Images With Entropy For Frugal Knowledge Distillation](https://ieeexplore.ieee.org/document/10878975), published in the IEEE Access. It contains the main methodology implementation script that can be used to reproduce the work as described in the paper.

<img src="https://github.com/michaelkinnas/Selecting-Images-with-Entropy-for-Frugal-Knowledge-Distillation/blob/main/overview.png?raw=true" width="100%">

## Prerequisites

- Python 3
- Pytorch >= 2.1.0
- Torchvision >= 0.16.0

## Preparation

To start with the experiments you begin by cloning the repository:

```bash
git clone https://github.com/michaelkinnas/Selecting-Images-with-Entropy-for-Frugal-Knowledge-Distillation.git
cd Selecting-Images-with-Entropy-for-Frugal-Knowledge-Distillation
```

To install the depedencies run:

```bash
pip install -r requirements.txt
```

It is recommended to use a python virtual environment:

- with python venv:

```bash
python -m venv <path-to-environment>/<environment-name>
source  <path-to-environment>/<environment-name>/bin/activate
pip install -r requirements.txt
```

- with conda:

```bash
conda create --name <environment-name> --file requirements.txt
```

## Main script

The `main.py` script executes the primary methodology steps outlined in the paper.  

The pipeline consists of four major steps:  

1. **Method** – The main sample selection approach:  
   - **top_n**: Selects the top N samples from the dataset.  
   - **top_n_pc**: Selects the top N samples per class.  
   - **km**: Uses KMeans clustering for sample selection.  
   - **km_pc**: Applies KMeans clustering within each class.  
   - **manifold_learning**: A state-of-the-art selection method used for comparison.  

2. **Representations** – The image representation technique used for evaluation:  
   - Grayscale histograms  
   - Average adjusted histograms  
   - RGB histograms  
   - Compressed feature vectors  
   - Logits vectors  
   - Patch entropy vectors  

3. **Evaluation** – Computes an importance score based on the selected representation:  
   - **Entropy** Our recommended criterion.  
   - **Variance**  

4. **Selection** – Determines sample selection priority based on the evaluation score:  
   - **Highest score**: Selects the highest-scoring samples first.  
   - **Lowest score**: Selects the lowest-scoring samples first.  
   - **Random**: Ignores evaluation scores and selects samples randomly.  

You can combine one option from each step to configure an experiment. Additionally, a few standard parameters, such as the dataset and the number of samples, are required, as described below.  


To run it use the command `python3 main.py` with the following parameters.

```
--dataset {cifar10,mnist,fashionmnist}
--method {top_n,top_n_per_category,kmeans,kmeans_per_category,manifold_learning,tn,tnpc,km,kmpc,ml}
                    Which overall sample selection method to use. You can use full name or abbreviated version
--n-clusters N_CLUSTERS
                    Optional. Only relevant if clustering method that receives number of clusters as argument is selected. If left as `None` an
                    automatic algorithm for determining optimal number of clusters will be used, such as silhouette score. Warning: slow.
--representations {histograms_grayscale,average_adjusted_histograms,histograms_rgb,compressed_feature_vectors,logits_vectors,patch_entropy_vectors,hg,aah,hrgb,cfv,lv,pev}
                    Used for the type of image representation method with normal and clustering methods.
--evaluation {entropy,variance}
                    Which evaluation method to use for the image scores. Default is entropy for which a representation is required.
--selection {highest_score,lowest_score,random,hs,ls,rng}
                    The sample selection criterion based upon the calculated score (entropy or other)
--n-samples N_SAMPLES
                    The number of samples to use
--size SIZE
--batch-size BATCH_SIZE
--total-epochs TOTAL_EPOCHS
--lr LR               learning rate
--use-validation-step
                    If set, a validation step will take place after each training epoch step.
--evaluate            If set, an evaluation will take place after the end of training.
--device {cuda,cpu}   The device to use
--seed SEED
--report {stdout,file}
--root-password ROOT_PASSWORD
                    If provided the script will command the computer to shutdown when finished.
```

### Examples of use

For example to run an experiment using the top 25000 samples based on grayscale histogram entropy evaluation you will run:


```bash
python main.py --dataset cifar10 --method top_n_per_category --representations histograms_grayscale --evaluation entropy --selection highest_score --n-samples 25000
```

## Acknowledgements

This work was funded by the European Union’s Horizon Europe research and innovation program under grant agreement No. 101120237 (ELIAS)

## License

This project is licensed under the Apache License 2.0. You can red the [LICENSE](https://github.com/michaelkinnas/Selecting-Images-with-Entropy-for-Frugal-Knowledge-Distillation/blob/main/LICENSE) file for more details.
