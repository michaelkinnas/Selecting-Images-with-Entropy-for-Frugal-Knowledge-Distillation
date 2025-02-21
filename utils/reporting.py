from datetime import datetime, timedelta

def report_to_file(file_path, args, n_clusters, trainset_subset, process_start_time, process_end_time, 
                       acc=None, precision=None, recall=None, f1=None, val_loss=None):

    f = open(file_path, "a")
    f.write(f"Dataset: {args.dataset}\n")
    f.write(f"Method: {args.method}\n")
    if args.method in ['kmeans', 'kmeans_per_category']: 
        if args.n_dimensions is not None:
            f.write(f"Dimensionality reduction components: {args.n_dimensions}\n")
        if args.n_clusters is None:
            f.write(f"Number of clusters calculated with silhoutete: {n_clusters}\n")
        else:
            f.write(f"Number of clusters: {n_clusters}\n")

    f.write(f"Representations: {args.representations}\n")
    f.write(f"Selection: {args.selection}\n") 
    f.write(f"Number of samples: {len(trainset_subset)}\n")
    f.write(f"Epochs: {args.total_epochs}\n")
    f.write(f"Started on {datetime.fromtimestamp(process_start_time)}\n")
    f.write(f"Finished on {datetime.fromtimestamp(process_end_time)}\n")
    f.write(f"Total elapsed time: {timedelta(seconds=process_end_time - process_start_time)} sec\n")
    
    if args.evaluate:
        f.write(f"Acc: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"Loss: {val_loss:.4f}\n")
    
    f.write(f"Energy: \n")
    f.write(f"Time measured: \n")
    f.write(f"Corrected energy: \n\n")
    f.close()


def report_to_std_out(args, n_clusters, trainset_subset, process_start_time, process_end_time, 
                       acc=None, precision=None, recall=None, f1=None, val_loss=None):
    
    repr_abr = {'hg':'histograms_grayscale',
                   'aah':'average_adjusted_histograms',
                   'hrgb':'histograms_rgb',
                   'cfv':'compressed_feature_vectors',
                   'lv':'logits_vectors'}

    methods_abr = {'km':'kmeans',
                   'kmpc':'kmeans_per_category',
                   'tnpc':'top_n_per_category',
                   'ml':'manifold_learning',
                   'hv':'highest_variance'}
    


    print(f"Dataset: {args.dataset}")

    if args.method in methods_abr:
        print(f"Method: {methods_abr[args.method]}")
    else:
        print(f"Method: {args.method}")
    
    if args.method in ['kmeans', 'kmeans_per_category', 'km', 'kmpc']:
        if args.n_dimensions is not None:
            print(f"Dimensionality reduction components: {args.n_dimensions}")
        elif args.n_clusters is None:
            print(f"Number of clusters calculated with silhoutete: {n_clusters}")
        else:
            print(f"Number of clusters: {n_clusters}")
    
    if args.representations in repr_abr:
        print(f"Representations: {repr_abr[args.representations]}")
    else:
        print(f"Representations: {args.representations}")
    
    print(f"Selection: {args.selection}") 
    print(f"Number of samples: {len(trainset_subset)}")
    print(f"Epochs: {args.total_epochs}")
    print(f"Started on {datetime.fromtimestamp(process_start_time)}")
    print(f"Finished on {datetime.fromtimestamp(process_end_time)}")
    print(f"Total elapsed time: {timedelta(seconds=process_end_time - process_start_time)} sec")
    
    if args.evaluate:
        print(f"Acc: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"Loss: {val_loss:.4f}")
    
    print(f"Energy: ")
    print(f"Time measured: ")
    print(f"Corrected energy: \n")