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

After converting segmentations to triangulated meshes, we manually cut out inner and outer surface and runs LDDMM to register the inner surface to the outer surface with the additional constraint that the deformation is restricted to the surface normal direction. We used the lengths of the flow lines, provided by each vertex trajectory under the transformation, to define the vertex-wise thickness between the cortical boundaries. The median of the distribution of the vertex-wise thickness was used to attribute a scalar thickness measurement to each ERC/TEC volume. Surfaces from ADNI are generated from manual segmentation, with only left hemisphere; surfaces from BIOCARD come from nnUNet prediction, with Left and right hemispheres cut and gnerated separately. 


The code for ERC/TEC surface reconstruction is available in this repository: [MeshLDDMMQP](https://github.com/kstouff4/MeshLDDMMQP). Pseudo codes:
```
bash Codes/ThicknessCalculations/runSurfaceRegistration.sh -s 1 -i Data/ADNI_Surface/cutted_surfaces/ -o Data/ADNI_Surface/mappedsurfaces/ -f all 
```
For population template generation,
```
python3 Codes/ThicknessCalculations/py-lddmm2/populationMappings.py
```

## Subregional Amygdalar Atrophy Coupled to High-Field Atlasing

To compute volumetric atrophy within amygdala subregions, we employed a two-stage diffeomorphic mapping procedure from the high-field template to each subject’s 3T segmentation. For each subject, all available time points were jointly aligned in a global LDDMM space using particle based varifold LDDMM from the amygdala, ERC, TEC, and hippocampus. A subject-specific midpoint template was then generated by computing the average initial momentum across time points and deforming the high field atlas. This subject-specific template was subsequently mapped to each individual time point using volume LDDMM.

Following deformation, voxel-wise segmentation masks were reconstructed from the particle representation by interpolating particle-defined anatomical boundaries onto a regular grid and assigning each voxel to the subregion with the highest empirical likelihood. Atrophy rates were computed from the slope of the fitted longitudinal model to the structures.

The code for amygdala subnuclei mapping is available in this repository: [projective-lddmm](https://github.com/kstouff4/projective-lddmm). Psudo codes:


To convert low-field segmentation into particles,

```
python3 Codes/amygala_subnuclei_analysis/makeLFsegParticles.py
```

To convert high-field segmentation into particles (at original resolution or 5x downsampled resolution):
```
python3 Codes/amygala_subnuclei_analysis/makeHFsehParticles.py
```
Stage 1 mapping with three regions (amygdala, ERC/TEC, Hippocampus) from template to all timepoints:
```
python3 Codes/amygala_subnuclei_analysis/framework_script_HF2LF.py
```
Calculate the average momentum and generate subject-specific template within mapped time points:
```
python3 Codes/amygala_subnuclei_analysis/templateGeneration.py
```
Separate the three regions into three particle files:
```
python3 Codes/amygala_subnuclei_analysis/splitParticles.py
```
Stage 2 mapping where each region is mapped from the subject-specific template to each time point individually:
```
python3 Codes/amygala_subnuclei_analysis/framework_script_HF2LF_Individual.py
```
Transform particles into segmentation by Nearest Neighbour or Gaussian interpolation:
```
Codes/amygala_subnuclei_analysis/makeParticlesToImages.py
```