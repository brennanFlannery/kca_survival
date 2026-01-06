import os
import argparse
from monai.networks.nets import resnet18, DenseNet169
import monai.transforms
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from DeepSurvImg import DeepSurv, create_dynamic_mlp
from DeepSurvImg import NegativeLogLikelihood, CoxLoss
from utils import SurvivalDatasetImgClinical, SurvivalDatasetImgFromFolder
from utils import c_index
from utils import adjust_learning_rate
from torch.utils.data import WeightedRandomSampler
from torchsummary import summary
import torchio as tio
from lifelines.statistics import logrank_test
from tqdm import tqdm
from monai.transforms import RandHistogramShift, RandAffine

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train DeepSurv models for survival analysis with optional data augmentation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--datafile', type=str, required=True,
                        help='Path to the h5 data file')
    parser.add_argument('--modelname', type=str, required=True,
                        help='Base name for the model (will be appended with model number)')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save trained models and metrics')
    
    # Training hyperparameters
    parser.add_argument('--num_models', type=int, default=50,
                        help='Number of models to train')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum number of training epochs')
    parser.add_argument('--lr_0', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=1e-4,
                        help='Learning rate decay')
    parser.add_argument('--weight_decay', type=float, default=0.000,
                        help='Weight decay for optimizer')
    parser.add_argument('--val_batchsize', type=int, default=4,
                        help='Batch size for validation')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (epochs)')
    
    # Model architecture
    parser.add_argument('--modeltype', type=str, default='Densenet',
                        choices=['Densenet', 'Resnet'],
                        help='Type of model architecture')
    parser.add_argument('--pre_trained_model', type=str, default=None,
                        help='Path to pre-trained model weights (optional)')
    parser.add_argument('--pre_trained_model_outputdim', type=int, default=5,
                        help='Output dimension of pre-trained model')
    parser.add_argument('--pre_train_freeze', action='store_true', default=True,
                        help='Freeze pre-trained model layers')
    parser.add_argument('--no_pre_train_freeze', action='store_false', dest='pre_train_freeze',
                        help='Do not freeze pre-trained model layers')
    parser.add_argument('--pre_train_mlp_depth', type=int, default=4,
                        help='Depth of MLP when using frozen pre-trained model')
    
    # Reproducibility and standardization
    parser.add_argument('--model_seed', type=int, default=42,
                        help='Random seed for model initialization')
    parser.add_argument('--data_seed', type=int, default=42,
                        help='Random seed for data ordering')
    parser.add_argument('--model_init_standardize', action='store_true', default=True,
                        help='Use fixed seed for model initialization')
    parser.add_argument('--no_model_init_standardize', action='store_false', dest='model_init_standardize',
                        help='Do not use fixed seed for model initialization')
    parser.add_argument('--data_order_standardize', action='store_true', default=False,
                        help='Use fixed seed for data loader ordering')
    parser.add_argument('--no_data_order_standardize', action='store_false', dest='data_order_standardize',
                        help='Do not use fixed seed for data loader ordering')
    parser.add_argument('--transform_standardize', action='store_true', default=True,
                        help='Disable all augmentations (master switch)')
    parser.add_argument('--no_transform_standardize', action='store_false', dest='transform_standardize',
                        help='Enable augmentations based on individual settings')
    
    # Augmentation settings
    parser.add_argument('--histnorm', type=str, default='none',
                        choices=['none', 'low', 'medium', 'high'],
                        help='Histogram shift intensity: low=10, medium=7, high=3 control points')
    parser.add_argument('--noise', type=str, default='none',
                        choices=['none', 'low', 'medium', 'high'],
                        help='Gaussian noise intensity: low=10, medium=25, high=50 std')
    parser.add_argument('--affine', type=str, default='none',
                        choices=['none', 'low', 'medium', 'high'],
                        help='Affine transform intensity: low=π/12,0.05,0.05; medium=π/6,0.15,0.15; high=π/4,0.3,0.3')
    
    # Evaluation phases
    parser.add_argument('--exval_phases', type=str, nargs='+', default=['train', 'test', 'val', 'exval'],
                        help='Phases to evaluate on')
    
    # Model trimming
    parser.add_argument('--trim_models', action='store_true', default=False,
                        help='Keep only best, worst, and middle models; delete all others to save disk space')
    
    return parser.parse_args()


def train(train_loader, test_loader, test_dataset, modelnum, args, device):

    # Basic model parameters and initializations
    model_name = args.modelname + f"_{modelnum}.pth"
    if args.model_init_standardize:
        torch.manual_seed(args.model_seed)
    if args.pre_trained_model is not None:
        model = resnet18(pretrained=False, num_classes=args.pre_trained_model_outputdim, n_input_channels=1, spatial_dims=3).to(device)
        model.load_state_dict({key.replace("module.", ""): value for key, value in torch.load(args.pre_trained_model)["model"].items()})
        if args.pre_train_freeze:
            # Freeze all layers
            for param in model.parameters():
                param.requires_grad = False
            # model.fc = torch.nn.Sequential(torch.nn.Linear(in_features=model.fc.in_features, out_features=1),
            #                                torch.nn.Tanh())
            model.fc = create_dynamic_mlp(args.pre_train_mlp_depth,input_dim=model.fc.in_features,output_dim=1, use_tanh=True)
        else:
            model.fc = torch.nn.Sequential(torch.nn.Linear(in_features=model.fc.in_features, out_features=1),torch.nn.Tanh())
        model.to(device)
    else:
        model = DeepSurv({"activation": "ReLu", "modeltype":"3D"}, modeltype=args.modeltype).to(device)
    criterion = CoxLoss({"l1_reg": 0, "l2_reg": 0}).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_0, weight_decay=args.weight_decay)

    # initialize values to track during training
    best_c_index = 0
    best_val_loss = np.inf
    val_loss = []
    t_loss = []
    val_c = []
    t_c = []
    flag = 0
    best_model_state_dict = None
    for epoch in range(1, args.epochs + 1):
        lr = adjust_learning_rate(optimizer, epoch,
                                  args.lr_0,
                                  args.lr_decay)
        # train step
        model.train()
        for X, y, e, pname in train_loader:
            if torch.sum(e) > 0:
                X = X.type('torch.FloatTensor').to(device)
                y = y.to(device)
                e = e.to(device)
                risk_pred = model(X)
                train_loss = criterion(risk_pred, y, e, model)
                train_c = c_index(-risk_pred, y, e)
                # updates parameters
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

        # valid step
        model.eval()
        count = 0
        final_pred = torch.zeros((len(test_dataset), 1)).to(device)
        final_y = torch.zeros((len(test_dataset), 1)).to(device)
        final_e = torch.zeros((len(test_dataset), 1)).to(device)
        for X, y, e, pname in test_loader:
            X = X.type('torch.FloatTensor').to(device)
            y = y.to(device)
            final_y[(count * args.val_batchsize):(count * args.val_batchsize + args.val_batchsize), :] = y  # y[::, None]
            e = e.to(device)
            final_e[(count * args.val_batchsize):(count * args.val_batchsize + args.val_batchsize), :] = e  # e[::, None]
            # makes predictions
            with torch.no_grad():
                risk_pred = model(X)
                optimizer.zero_grad()
                final_pred[(count * args.val_batchsize):(count * args.val_batchsize + args.val_batchsize), :] = risk_pred
            count += 1

        valid_loss = criterion(final_pred, final_y, final_e, model)
        valid_c = c_index(-final_pred, final_y, final_e)

        flag +=1

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_c_index = valid_c
            saved_epoch = epoch
            flag = 0
            # saves the best model
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch}, os.path.join(args.save_dir, model_name))
            best_model_state_dict = model.state_dict()


        # notes that, train loader and valid loader both have one batch!!!
        val_loss.append(valid_loss.item())
        t_loss.append(train_loss.item())
        val_c.append(valid_c)
        t_c.append(train_c)

        if isinstance(args.patience, int):
            if flag > args.patience:
                model.load_state_dict(best_model_state_dict)
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch}, os.path.join(args.save_dir, model_name))
                return model, best_c_index

        # print('\rEpoch: {}\tLoss: {:.8f}({:.8f})\tc-index: {:.8f}({:.8f})\tlr: {:g}'.format(
        #     epoch, train_loss.item(), valid_loss.item(), train_c, valid_c, lr), end='', flush=False)

    # Reload the best model, THIS WAS DONE EVERY EPOCH LAST TIME, WAS THAT BETTER?
    model.load_state_dict(best_model_state_dict)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch}, os.path.join(args.save_dir, model_name))

    return model, best_c_index

def compute_c_index(data):
    c = c_index(-np.array(data["risk_score"]),
                np.array(data["t"]),
                np.array(data["e"]))
    return c

def compute_hazard_ratio(data, thresh):
    data = pd.DataFrame(data)
    data["risk_group"] = [1 if x > thresh else 0 for x in data["risk_score"]]
    results = logrank_test(data[data["risk_group"] == 0]["t"].to_numpy(),
                           data[data["risk_group"] == 1]["t"].to_numpy(),
                           event_observed_A=data[data["risk_group"] == 0]["e"].to_numpy(),
                           event_observed_B=data[data["risk_group"] == 1]["e"].to_numpy())
    return results.test_statistic, results.p_value

def find_best_threshold(model_outputs):
    thresholds = np.linspace(start=np.quantile(model_outputs["risk_score"], 0.15),
                             stop=np.quantile(model_outputs["risk_score"], 0.85), num=100)
    hrs = [compute_hazard_ratio(model_outputs, thresh)[0] for thresh in thresholds]
    best_hr = np.max(hrs)
    best_thresh = thresholds[np.argmax(hrs)]

    return best_thresh, best_hr

def run_model(model, loader, dev = None):
    model_outputs = {"risk_score": [], "e": [], "t": [], "name":[]}
    model.eval()
    # Run model on all data in the phase
    for (X, y, e, pnames) in loader:
        X = X.type('torch.FloatTensor').to(dev)
        with torch.no_grad():
            risk_pred = model(X)
            model_outputs["risk_score"].append(risk_pred.squeeze().item())
            model_outputs["e"].append(e.item())
            model_outputs["t"].append(y.item())
            model_outputs["name"].append(pnames[0].decode())

    return model_outputs

def validate(model, modelnum, dev, args):
    model_name = args.modelname + f"_{modelnum}.pth"
    best_threshold = 0
    outdata = {}
    outdata["modelname"] = model_name
    for phase in args.exval_phases:

        # Set up datasets
        dataset = SurvivalDatasetImgClinical(h5_file=args.datafile, split=phase, transform=None, is_vol=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        # Get risk scores
        model_outputs = run_model(model, loader, dev)

        # Find best threshold on training set
        if phase == "train":
            best_threshold, best_hr = find_best_threshold(model_outputs)

        hr, p = compute_hazard_ratio(model_outputs, np.median(model_outputs["risk_score"]))
        hr_mod, p_mod = compute_hazard_ratio(model_outputs, best_threshold)
        c = compute_c_index(model_outputs)

        # Store metrics
        outdata[f"{phase}_hr"] = [hr]
        outdata[f"{phase}_p"] = [p]
        outdata[f"{phase}_hr_mod"] = [hr_mod]
        outdata[f"{phase}_p_mod"] = [p_mod]
        outdata[f"{phase}_c"] = [c]

    return pd.DataFrame(outdata)


def get_transforms(histnorm_level, affine_level, noise_level):
    """
    Create image transforms with specified intensity levels.
    
    Args:
        histnorm_level: "none", "low", "medium", "high" for histogram shift
        affine_level: "none", "low", "medium", "high" for affine transforms
        noise_level: "none", "low", "medium", "high" for gaussian noise
    """
    transforms = []
    
    # Histogram shift parameters
    histnorm_params = {"low": 10, "medium": 7, "high": 3}
    if histnorm_level in histnorm_params:
        hist_transform = RandHistogramShift(
            num_control_points=histnorm_params[histnorm_level], 
            prob=0.5
        )
        transforms.append(hist_transform)
    
    # Affine transform parameters
    affine_params = {
        "low": {"rotate": np.pi / 12, "scale": 0.05, "shear": 0.05},
        "medium": {"rotate": np.pi / 6, "scale": 0.15, "shear": 0.15},
        "high": {"rotate": np.pi / 4, "scale": 0.30, "shear": 0.30}
    }
    if affine_level in affine_params:
        params = affine_params[affine_level]
        affine_transform = RandAffine(
            prob=0.5,
            rotate_range=(params["rotate"], params["rotate"], params["rotate"]),
            scale_range=(params["scale"], params["scale"], params["scale"]),
            shear_range=(params["shear"], params["shear"], params["shear"])
        )
        transforms.append(affine_transform)
    
    # Gaussian noise parameters
    noise_params = {"low": 10, "medium": 25, "high": 50}
    if noise_level in noise_params:
        noise_transform = monai.transforms.RandGaussianNoise(
            prob=0.5, 
            std=noise_params[noise_level]
        )
        transforms.append(noise_transform)

    img_transform = monai.transforms.Compose(transforms)
    return img_transform

def trim_models(model_metrics, args, metric='val_c'):
    """
    Keep only the best, worst, and middle performing models.
    Deletes all other model files to save disk space.
    
    Args:
        model_metrics: DataFrame with model performance metrics
        args: Command line arguments containing save_dir and modelname
        metric: Performance metric to use for ranking (default: 'val_c')
    
    Returns:
        DataFrame with only the kept models and an added 'kept' column
    """
    if len(model_metrics) < 3:
        print("Less than 3 models trained. Skipping trimming.")
        model_metrics['kept'] = 'yes'
        model_metrics['kept_reason'] = 'too_few_models'
        return model_metrics
    
    # Sort by the chosen metric (higher is better for c-index)
    sorted_metrics = model_metrics.sort_values(by=metric, ascending=False).reset_index(drop=True)
    
    # Identify which models to keep
    best_idx = 0  # Best performer
    worst_idx = len(sorted_metrics) - 1  # Worst performer
    middle_idx = len(sorted_metrics) // 2  # Middle performer
    
    keep_indices = {best_idx, worst_idx, middle_idx}
    
    # Extract model numbers from modelname column
    # Format is typically: modelname_0.pth, modelname_1.pth, etc.
    sorted_metrics['model_number'] = sorted_metrics['modelname'].str.extract(r'_(\d+)\.pth$').astype(int)
    
    # Mark which models to keep
    sorted_metrics['kept'] = ['no'] * len(sorted_metrics)
    sorted_metrics['kept_reason'] = ['deleted'] * len(sorted_metrics)
    
    sorted_metrics.loc[best_idx, 'kept'] = 'yes'
    sorted_metrics.loc[best_idx, 'kept_reason'] = 'best'
    
    sorted_metrics.loc[worst_idx, 'kept'] = 'yes'
    sorted_metrics.loc[worst_idx, 'kept_reason'] = 'worst'
    
    sorted_metrics.loc[middle_idx, 'kept'] = 'yes'
    sorted_metrics.loc[middle_idx, 'kept_reason'] = 'middle'
    
    print(f"\n{'='*60}")
    print("TRIMMING MODELS")
    print(f"{'='*60}")
    print(f"Ranking metric: {metric}")
    print(f"Total models: {len(sorted_metrics)}")
    print(f"\nKeeping 3 models:")
    print(f"  Best:   Model {sorted_metrics.loc[best_idx, 'model_number']} "
          f"({metric}={sorted_metrics.loc[best_idx, metric]:.4f})")
    print(f"  Middle: Model {sorted_metrics.loc[middle_idx, 'model_number']} "
          f"({metric}={sorted_metrics.loc[middle_idx, metric]:.4f})")
    print(f"  Worst:  Model {sorted_metrics.loc[worst_idx, 'model_number']} "
          f"({metric}={sorted_metrics.loc[worst_idx, metric]:.4f})")
    
    # Delete models not in keep list
    deleted_count = 0
    failed_deletions = []
    
    for idx, row in sorted_metrics.iterrows():
        if idx not in keep_indices:
            model_file = os.path.join(args.save_dir, row['modelname'])
            if os.path.exists(model_file):
                try:
                    os.remove(model_file)
                    deleted_count += 1
                except Exception as e:
                    failed_deletions.append((model_file, str(e)))
                    print(f"Warning: Failed to delete {model_file}: {e}")
    
    print(f"\nDeleted {deleted_count} model files")
    if failed_deletions:
        print(f"Failed to delete {len(failed_deletions)} files")
    
    # Calculate space saved (approximate)
    if deleted_count > 0:
        # Get size of one of the kept models as reference
        kept_model_file = os.path.join(args.save_dir, 
                                       sorted_metrics.loc[best_idx, 'modelname'])
        if os.path.exists(kept_model_file):
            model_size_mb = os.path.getsize(kept_model_file) / (1024 * 1024)
            space_saved_mb = model_size_mb * deleted_count
            print(f"Approximate space saved: {space_saved_mb:.1f} MB "
                  f"({space_saved_mb/1024:.2f} GB)")
    
    print(f"{'='*60}\n")
    
    return sorted_metrics

def make_histogram(data, name):
    # Create the histogram with 20 bins
    plt.hist(data, bins=20, edgecolor='black')

    # Add titles and labels
    plt.title(name)
    plt.xlabel(f'{name} values')
    plt.ylabel('frequency')

    # Show the plot
    plt.show()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set up transforms based on arguments
    if args.transform_standardize:
        img_transform = None
    else:
        img_transform = get_transforms(args.histnorm, args.affine, args.noise)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset = SurvivalDatasetImgClinical(h5_file=args.datafile, split="train", transform=img_transform,
                                       is_vol=True)
    test_dataset = SurvivalDatasetImgClinical(h5_file=args.datafile, split="test", transform=None,
                                      is_vol=True)

    # Setup our weighted random sampler to ensure there are event cases in every batch
    events = [x[0] for x in train_dataset.e]
    weights = 1 - np.array([train_dataset.__len__() - train_dataset.num_events, train_dataset.num_events]) / train_dataset.__len__()
    sample_weights = [weights[i] for i in events]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=train_dataset.__len__(), replacement=True)

    # Create dataloaders
    if args.data_order_standardize:
        torch.manual_seed(args.data_seed)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=sampler, drop_last=True)
    if args.data_order_standardize:
        torch.manual_seed(args.data_seed)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_batchsize, shuffle=True)

    # Train models and get metrics
    model_metrics = []
    for i in range(args.num_models):
        print(f"\n{'='*60}")
        print(f"Training model {i+1}/{args.num_models}")
        print(f"{'='*60}")
        
        best_model, best_c = train(train_loader, test_loader, test_dataset, i, args, device)
        metrics = validate(best_model, i, device, args)
        model_metrics = pd.concat([model_metrics, metrics], ignore_index=True) if len(model_metrics) > 0 else metrics

        print(f"Model {i} completed - C-index: {best_c:.4f}")

    # Trim models if requested (keep only best, worst, and middle)
    if args.trim_models:
        model_metrics = trim_models(model_metrics, args, metric='val_c')
    
    # Make graphs of metric distributions
    print(f"\n{'='*60}")
    print("Generating metric histograms...")
    print(f"{'='*60}")
    
    # Only plot metrics that are numeric (skip 'kept' and 'kept_reason' if present)
    numeric_cols = [col for col in model_metrics.columns 
                    if col not in ['modelname', 'kept', 'kept_reason', 'model_number']]
    for metric in numeric_cols:
        make_histogram(model_metrics[metric].values, metric)

    # Save model metrics
    metrics_path = os.path.join(args.save_dir, "model_metrics.xlsx")
    model_metrics.to_excel(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")
    
    if args.trim_models:
        print(f"\nNote: Model trimming was enabled. Only best, worst, and middle models were kept.")
        kept_models = model_metrics[model_metrics['kept'] == 'yes']['modelname'].tolist()
        print(f"Kept models: {', '.join(kept_models)}")


