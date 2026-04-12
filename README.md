# MTL_xIV_lddmm_analysis
# Rostral Associations of MRI Atrophy of the Amygdala and Entorhinal Cortex Across the AD Spectrum

This repository contains code for the paper "Rostral Associations of MRI Atrophy of the Amygdala and Entorhinal Cortex Across the AD Spectrum" (https://doi.org/10.64898/2026.01.27.26344987). It was developed to examine associations between atrophy in the amygdala, entorhinal cortex, and hippocampus, based on magnetic resonance imaging (MRI) and Positron Emission Tomography (PET) from two independent cohorts: the Alzheimer’s Disease Neuroimaging Initiative (ADNI) and the Biomarkers of Cognitive Decline Among Normal Individuals (BIOCARD) study. It includes longitudinal analysis of volume atrophy in the amygdala, ERC/TEC, and hippocampus, ERC/TEC cortical thickness reconstruction, amygdala subnuclei analysis, and a correlation study between Tau PET SUVR and volume atrophy rate.

## Data Availability
This repository does not include any imaging data from ADNI or BIOCARD due to data usage agreements. These datasets are publicly available but require registration and approval through their respective portals.


## Imaging data

Participants in the BIOCARD study were categorized into four diagnostic groups based on longitudinal clinical evaluations: cognitively normal (CN), impaired not MCI, MCI, and AD. Subjects in all groups were required to have at least three 3T MR scans for longitudinal comparison. Demographic summaries, including baseline age, scan count, and follow-up duration, are provided below:

| Dataset         | Metric               | Normal       | Impaired (not MCI) | MCI          | Dementia     |
| --------------- | -------------------- | ------------ | ------------------ | ------------ | ------------ |
| BIOCARD (n=130) | # Subjects           | 51           | 40                 | 31           | 8            |
|                 | Baseline Age (years) | 67.59 ± 6.74 | 68.68 ± 6.08       | 70.29 ± 7.86 | 70.63 ± 6.54 |
|                 | # Scans              | 3.73 ± 0.60  | 3.80 ± 0.78        | 3.68 ± 0.69  | 3.38 ± 0.48  |
|                 | Scan Range           | 6.67 ± 1.29  | 6.90 ± 1.55        | 6.55 ± 1.60  | 5.75 ± 1.64  |

Longitudinal 1.5 T structural MRI data acquired with both sagittal and coronal slice orientations were additionally used to validate the robustness of the estimated atrophy-rate patterns. Sagittal acquisitions had a voxel resolution of $1.5 \times 0.9375 \times 0.9375\space mm^3$, while coronal acquisitions had a voxel resolution of $0.9375 \times 2.0 \times 0.9375\space mm^3$. Detailed acquisition parameters for these scans have been previously reported [here](https://pubmed.ncbi.nlm.nih.gov/25101236/).

Diagnoses of the participants in the ADNI dataset are grouped into  Control, which contains CN and SMC (Subjective Memory Concern); MCI, which consists of EMCI (Early Mild Cognitive Impairment), MCI and LMCI (Late Mild Cognitive Impairment); and AD, as shown below. Subjects were required to have at least three 3T MR scans for longitudinal comparison.


| Dataset             | Metric               | CN           | SMC          | EMCI         | MCI          | LMCI         | AD           |
| ------------------- | -------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| ADNI                | # Subjects           | 1224         | 84           | 303          | 954          | 168          | 485          |
|                     | Baseline Age (years) | 70.46 ± 7.11 | 72.72 ± 5.75 | 71.40 ± 7.38 | 71.69 ± 7.49 | 72.55 ± 7.91 | 74.50 ± 8.46 |
|                     | # Scans              | 4.90 ± 2.05  | 4.12 ± 1.09  | 5.73 ± 1.89  | 4.90 ± 2.29  | 5.06 ± 1.63  | 3.74 ± 0.70  |
|                     | Scan Range           | 5.69 ± 3.45  | 4.01 ± 2.28  | 3.95 ± 2.70  | 4.87 ± 3.43  | 2.95 ± 2.16  | 1.52 ± 0.91  |

## MTL segmentation

Automated segmentation of five medial temporal lobe (MTL) structures - Amygdala, ERC, TEC, Anterior Hippocampus (HA), and Posterior Hippocampus (HP) – was performed on T1-weighted 3T MRI using a combination of deep learning and atlas-based methods. Specifically, ASHS was used to delineate HA and HP subfields. Combined with manual segmentations of amygdala, ERC, and TEC, such delineations served as ground truth for training.

Segmented images were used in developing a self-configuring deep convolutional neural network called nnU-Net. nnUNet automatically configures the entire segmentation pipeline — including preprocessing, network architecture, training schedule, data augmentation, and postprocessing — based solely on the properties of the training dataset. It builds upon a 3D U-Net backbone but adapts key hyperparameters and architectural components to match the data characteristics, enabling state-of-the-art performance without manual tuning. 

For MTL label segmentation, the 3D full-resolution configuration used a $112\times128\times160$ patch size with batch size=2 at $1.2\times1\times1$ mm$^3$. The manual segmentation was partitioned into training and testing subsets using a $90:10$ split. The training objective combined focal loss, cross-entropy loss, and Dice loss into a unified loss function for optimization. To enable dual-hemisphere prediction, flip augmentation was disabled during training, and all manual segmentations were standardized to a single hemisphere (left hemisphere for ADNI and right hemisphere for BIOCARD). During inference, both the original and left–right flipped images were processed by the model. The resulting segmentations were then mapped back and fused to obtain a dual-hemisphere medial temporal lobe (MTL) segmentation. The held-out test set was used exclusively for final performance assessment with baseline methods, including MRICloud, FreeSurfer, and ASHS. For the training and validation using nnUNet, click [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) for more information. 

|                              | ADNI | BIOCARD |
| ---------------------------- | ---- | ------- |
| # Scans                      | 294  | 371     |
| # Subjects                   | 67   | 107     |
| # of time points per subject | 4.39 | 3.47    |


## Atrophy calculation

Atrophy rates were calculated for each subject using longitudinal volume estimates derived from region-specific segmentation masks. Volumes were computed by multiplying the number of voxels within each label by the voxel dimensions ($1.0 \times 1.0 \times 1.2 mm^3$). A linear regression model was fitted to each subject's longitudinal volume trajectory to estimate the annualized rate of change, expressed as both absolute volume change ($mm^3$/year) and percent change relative to baseline. Standard deviation of all predicted segmentation in each region (amygdala, ERC/TEC, hippocampus) is calculated, and time points whose regions exceed 2.5 the standard deviation from the group mean within the subject were labeled as outliers and excluded from subsequent analysis. Visual inspection was also performed to remove outliers due to poor image quality.

## Entorhinal Cortex Reconstruction and Cortical Thickness Calculation

Following segmentation, volumetric labels were converted into triangulated surface meshes. For each subject, the inner and outer cortical surfaces of the entorhinal and transentorhinal cortex (ERC/TEC) were manually extracted. A diffeomorphic registration was then performed using Large Deformation Diffeomorphic Metric Mapping (LDDMM), aligning the inner surface to the outer surface under the constraint that deformation occurs strictly along the surface normal direction.

Cortical thickness was defined vertex-wise as the length of the deformation trajectories (flow lines) generated during the LDDMM mapping between the two surfaces. A scalar thickness value for each ERC/TEC region was subsequently obtained by taking the median of the vertex-wise thickness distribution.

Surface meshes for the ADNI dataset were derived from manual segmentations and include only the left hemisphere. In contrast, BIOCARD surfaces were generated from nnU-Net predictions, with left and right hemispheres processed independently following segmentation.


The code used for ERC/TEC surface reconstruction and thickness computation is publicly available at [MeshLDDMMQP](https://github.com/kstouff4/MeshLDDMMQP). A representative command for running the surface registration pipeline is:
```
bash Codes/ThicknessCalculations/runSurfaceRegistration.sh -s 1 -i Data/ADNI_Surface/cutted_surfaces/ -o Data/ADNI_Surface/mappedsurfaces/ -f all 
```
Population template generation is performed using:
```
python3 Codes/ThicknessCalculations/py-lddmm2/populationMappings.py
```

## Subregional Amygdalar Atrophy Coupled to High-Field Atlasing

To quantify volumetric atrophy within amygdala subregions, we employed a two-stage diffeomorphic mapping framework that transfers high-field atlas information to subject-specific 3T MRI segmentations.

### Stage 1: Joint Longitudinal Alignment and Template Construction:

For each subject, all available time points were jointly aligned within a common diffeomorphic space using particle-based varifold LDDMM. This registration incorporates multiple anatomical structures, including the amygdala, ERC/TEC, and hippocampus.

A subject-specific midpoint template was then constructed by averaging the initial momenta across all time points and applying the resulting deformation to the high-field atlas. This step provides a temporally unbiased representation of subject anatomy.

### Stage 2: Individual Timepoint Mapping:

The subject-specific template was subsequently mapped to each individual time point using volume-based LDDMM, one region at a time. This enables consistent propagation of subregional labels across time while preserving anatomical correspondence.

Following deformation, voxel-wise segmentation masks were reconstructed from the particle representation by interpolating particle-defined boundaries onto a regular grid. Each voxel was assigned to a subregion based on the highest empirical likelihood under the particle model.

Finally, atrophy rates were computed for each subregion by fitting longitudinal models and estimating the slope of volume change over time.

### Implementation Details:

The code full pipeline of particle LDDMM is available at [xIV-lddmm-Particle](https://github.com/kstouff4/xIV-LDDMM-Particle). A representative command for running the amygdala subnuclei analysis pipeline is:

Particle Generation:

Low-field segmentations:
```
python3 Codes/amygala_subnuclei_analysis/makeLFsegParticles.py
```

High-field segmentations (original or 5x downsampled resolution):
```
python3 Codes/amygala_subnuclei_analysis/makeHFsehParticles.py
```

Stage 1: Template-to-Timepoint Mapping (on amygdala, ERC/TEC, Hippocampus):
```
python3 Codes/amygala_subnuclei_analysis/framework_script_HF2LF.py
```

Subject-Specific Template Generation:
```
python3 Codes/amygala_subnuclei_analysis/templateGeneration.py
```

Region Separation:
```
python3 Codes/amygala_subnuclei_analysis/splitParticles.py
```

Stage 2: Individual Mapping:
```
python3 Codes/amygala_subnuclei_analysis/framework_script_HF2LF_Individual.py
```

Particle-to-Image Reconstruction:
```
Codes/amygala_subnuclei_analysis/makeParticlesToImages.py
```
Segmentation volumes are reconstructed using either nearest-neighbor assignment or Gaussian interpolation of particle-defined anatomical likelihoods.