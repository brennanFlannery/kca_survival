# KidneyRadiology - Required Files for Core Scripts

This README documents the core Python scripts and their required dependency files in the KidneyRadiology folder.

## Main Scripts

### h5_from_cohort_finder.py
Converts DICOM folders to H5 format by loading clinical data, normalizing images, using UNet to extract kidney/tumor ROIs, and saving processed volumes with survival data into H5 files organized by train/test/val splits.

### multi_train_stats_cli_all.py
Performs statistical analysis (ANOVA, Tukey HSD, Levene tests) and generates grouped boxplots comparing performance metrics across multiple ResNet, EfficientNet, and ViT models with different feature configurations.

### multitrain_deepsurv.py
Trains multiple DeepSurv survival analysis models with configurable augmentation settings, performs validation across multiple phases, computes C-index and hazard ratios, and optionally trims models to keep only best/worst/middle performers.

### segment_from_dicom.py
Processes DICOM folders by converting them to NIfTI files, running TotalSegmentator and UNet segmentation on kidneys, calculating 3D DICE scores to compare segmentations, and generating visualization overlays.

## Required Dependency Files

### unet.py
Implements the U-Net architecture for biomedical image segmentation with configurable depth, width factor, padding, batch normalization, and upsampling modes.

### DeepSurvImg.py
Defines DeepSurv neural network architectures for survival analysis (2D/3D ResNet, DenseNet), loss functions (CoxLoss, NegativeLogLikelihood), dynamic MLP creation utilities, and regularization classes.

### utils.py
Provides utility functions including survival analysis datasets (SurvivalDatasetImgClinical, SurvivalDatasetImgFromFolder), C-index calculation, learning rate adjustment, config file reading, and logging setup.

### resnet3d.py
Implements 3D ResNet architectures (BasicBlock, Bottleneck, ResNet) for processing volumetric medical imaging data with configurable depth and channel configurations.

### standardUtils.py
**Note:** This file is imported by `h5_from_cohort_finder.py` but was not found in the KidneyRadiology folder; it may be located in a parent directory or needs to be provided separately; contains `multi_slice_viewer` function (though unused in the current script).

