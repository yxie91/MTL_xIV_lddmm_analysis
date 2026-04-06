# MTL_xIV_lddmm_analysis
# Rostral Associations of MRI Atrophy of the Amygdala and Entorhinal Cortex Across the AD Spectrum

This repository contains code for the paper "Rostral Associations of MRI Atrophy of the Amygdala and Entorhinal Cortex Across the AD Spectrum" (https://doi.org/10.64898/2026.01.27.26344987). It was developed to examine associations between atrophy in the amygdala, entorhinal cortex, and hippocampus, based on magnetic resonance imaging (MRI) and Positron Emission Tomography (PET) from two independent cohorts: the Alzheimer’s Disease Neuroimaging Initiative (ADNI) and the Biomarkers of Cognitive Decline Among Normal Individuals (BIOCARD) study. It includes longitudinal analysis of volume atrophy in the amygdala, ERC/TEC, and hippocampus, ERC/TEC cortical thickness reconstruction, amygdala subnuclei analysis, and a correlation study between Tau PET SUVR and volume atrophy rate.

## Data Availability
This repository does not include any imaging data from ADNI or BIOCARD due to data usage agreements. These datasets are publicly available but require registration and approval through their respective portals.

## Setup
Please use environment.yml to install required packages.

Codes starting with *Fi_...* mean that Figure i in the paper is generated using the Python files. For example, 6x6 T1 collage is generated from *Code/F1_T1collage.py*; Figure 6 is generated from *Code/F6_LRVA_ADNI12GO.py* and *Code/F6_LRVA_BIOCARD.py*. Once you get access to the root directory, you can run the Python files and get the figures as in the papers. The visualization of the images and segmentations involves mainly two software: [ITK-SNAP](https://www.itksnap.org/pmwiki/pmwiki.php) and [PARAVIEW](https://www.paraview.org/).

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

The manual and predicted segmentations in BIOCARD dataset are summarized below.

|  Dataset   |                   Directory                   |                         Description                          |
| :--------: | :-------------------------------------------: | :----------------------------------------------------------: |
| BIOCARD 3T |        Data/Dataset006_axis0/imagesTr         |           Ground truth T1 images used for training           |
|            |        Data/Dataset006_axis0/labelsTr         |             Manual segmentation, Left Hemisphere             |
|            |     Data/Dataset006_axis0/imagesTs_mprage     |               test images used for prediction                |
|            |  Data/Dataset006_axis0/imagesTs_pred_mprage   |       Segmentation of the test images, left Hemisphere       |
|            |    Data/Dataset006_axis0/imagesTs_pred_RH     |      Segmentation of the test images, right Hemisphere       |
|            |   Data/Dataset006_axis0/imagesTs_pred_fuse    | Segmentation of the test images, left and right Hemisphere fused |
|            | Data/Sheets/control_SUBJECT_DOB_multiple.xlsx | Subject, Diagnosis and Date of birth (DOB) of control subjects |
|            | Data/Sheets/MCIDEM_SUBJECT_DOB_multiple.xlsx  | Subject, Diagnosis and Date of birth (DOB) of MCI/AD subjects |
|            |        Data/Sheets/SUBJECT_TP_PET.xlsx        |        Timepoints for each subject in format *yymmdd*        |

For 1.5T BIOCARD T1 images, the preprocessing resampled them into $1.2 \times 1 \times 1\ $mm$^3$, the resolution as 3T BIOCARD. And we use the model trained on 3T images to predict 1.5T data: 

|                 Dataset                  |                Directory                 |                         Description                          |
| :--------------------------------------: | :--------------------------------------: | :----------------------------------------------------------: |
|      BIOCARD 1.5T, sagittal slicing      |   Data/Dataset006_axis0/imagesTs_15mm    |               test images used for prediction                |
| $1.5 \times 0.9375 \times 0.9375$ mm$^3$ |   Data/Dataset006_axis0/labelsTs_15mm    |       Segmentation of the test images, left Hemisphere       |
|                                          |  Data/Dataset006_axis0/labelsTs_15mm_RH  |      Segmentation of the test images, right Hemisphere       |
|                                          | Data/Dataset006_axis0/labelsTs_15mm_fuse | Segmentation of the test images, left and right Hemisphere fused |
|                                          |     Data/Sheets/Demographic_15T.csv      |              Demographic data (DOB, Diagnosis)               |
|                                          | Data/Sheets/set2a_acquisition_dates.csv  |      Acquisition date for each scan in format *yymmdd*       |
|                                          |        Data/Sheets/15Outlier.xlsx        |         Scan outliers selected by visual validation          |

|                 Dataset                  |                Directory                 |                         Description                          |
| :--------------------------------------: | :--------------------------------------: | :----------------------------------------------------------: |
|      BIOCARD 1.5T, sagittal slicing      |   Data/Dataset006_axis0/imagesTs_20mm    |               test images used for prediction                |
| $0.9375 \times 2.0 \times 0.9375$ mm$^3$ |   Data/Dataset006_axis0/labelsTs_20mm    |       Segmentation of the test images, left Hemisphere       |
|                                          |  Data/Dataset006_axis0/labelsTs_20mm_RH  |      Segmentation of the test images, right Hemisphere       |
|                                          | Data/Dataset006_axis0/labelsTs_20mm_fuse | Segmentation of the test images, left and right Hemisphere fused |
|                                          |     Data/Sheets/Demographic_20T.csv      |              Demographic data (DOB, Diagnosis)               |
|                                          | Data/Sheets/set2b_acquisition_dates.csv  |      Acquisition date for each scan in format *yymmdd*       |
|                                          |        Data/Sheets/20Outlier.xlsx        |         Scan outliers selected by visual validation          |

The manual and predicted segmentations in ADNI dataset are summarized below.

|    Dataset     |               Directory               |                         Description                          |
| :------------: | :-----------------------------------: | :----------------------------------------------------------: |
| ADNI 1, 2 & GO |   Data/Dataset103_ADNIall/imagesTr    |           Ground truth T1 images used for training           |
|                |   Data/Dataset103_ADNIall/labelsTr    |             Manual segmentation, Left Hemisphere             |
|                |   Data/Dataset103_ADNIall/imagesTs    |               test images used for prediction                |
|                |   Data/Dataset103_ADNIall/labelsTs    |       Segmentation of the test images, left Hemisphere       |
|                |  Data/Dataset103_ADNIall/labelsTs_RH  |      Segmentation of the test images, right Hemisphere       |
|                | Data/Dataset103_ADNIall/labelsTs_fuse | Segmentation of the test images, left and right Hemisphere fused |
|                |  Data/Sheets/rawthk_v13_kms_con.mat   |  Subset of subject and age information for control subjects  |
|                |  Data/Sheets/rawthk_v13_kms_pre.mat   |  Subset of subject and age information for control subjects  |
|                |  Data/Sheets/rawthk_v13_kms_mci.mat   |    Subset of subject and age information for MCI subjects    |

For 3T ADNI 3, 4 T1 images, the preprocessing resampled them into $1.2 \times 1 \times 1\ $mm$^3$, the resolution as 3T images in ADNI 1, 2 & GO. And we use the same model to predict ADNI 3, 4 data: 

|  Dataset  |               Directory                |                          Decription                          |
| :-------: | :------------------------------------: | :----------------------------------------------------------: |
| ADNI 3, 4 |    Data/Dataset102_ADNI34/imagesTs     |               test images used for prediction                |
|           |    Data/Dataset102_ADNI34/labelsTs     |       Segmentation of the test images, left Hemisphere       |
|           |   Data/Dataset102_ADNI34/labelsTs_RH   |      Segmentation of the test images, right Hemisphere       |
|           |  Data/Dataset102_ADNI34/labelsTs_fuse  | Segmentation of the test images, left and right Hemisphere fused |
|           | Data/Sheets/ADNI34_MPRAGE_metadata.csv | Demographic data (disease group, sex age, acquisition date)  |
|           |     Data/Sheets/ADNI34Outlier.xlsx     |         Scan outliers selected by visual validation          |

For Figure 5, use *Code/F5_overlap_seg.py* to generate the combined version manual and predicted segmentation, we can use ITK-SNAP to visualize with color code *Data/Sheets/itksnap_colorcode.txt*. 

MRICloud segmentation of ADNI and BIOCARD data can be found below:

|       Directory       |                   Decription                   |
| :-------------------: | :--------------------------------------------: |
| Data/MRICloud_ADNI34  | MRICloud segmentation of a subset in ADNI 3, 4 |
| Data/MRICloud_BIOCARD |  MRICloud segmentation of a subset in BIOCARD  |

## Atrophy calculation

Atrophy rates were calculated for each subject using longitudinal volume estimates derived from region-specific segmentation masks. Volumes were computed by multiplying the number of voxels within each label by the voxel dimensions ($1.0 \times 1.0 \times 1.2$ mm). A linear regression model was fitted to each subject's longitudinal volume trajectory to estimate the annualized rate of change, expressed as both absolute volume change (mm$^3$/year) and percent change relative to baseline. Standard deviation of all predicted segmentation in each region (amygdala, ERC/TEC, hippocampus) is calculated, and time points whose regions exceed 2.5 the standard deviation from the group mean within the subject were labeled as outliers and excluded from subsequent analysis. Visual inspection was also performed to remove outliers due to poor image quality.

Figure 7, 8, 9 is generated using python files below:

| Figure |       Code directory       |                          Decription                          |
| :----: | :------------------------: | :----------------------------------------------------------: |
|   7    |  Code/F7_BIOCARD_VAage.py  | Figure 7 left panel, atrophy rate and age in control and MCI/AD groups in BIOCARD |
|        | Code/F7_BIOCARD_VA_315T.py | Figure 7 right panel, atrophy rate in datasets of different scan resolutions in BIOCARD |
|        |       Code/F7_bar.py       |                        Bar plot code                         |
|   8    | Code/F8_ADNI12GO_VA_2g.py  | Figure 8 left panel, atrophy rate and age in control and MCI/AD groups in ADNI 1, 2 & GO |
|        |  Code/F8_ADNI34_VA_3g.py   | Figure 8 right panel, atrophy rate in control, MCI, and AD groups in ADNI 3, 4 |
|        |       Code/F8_bar.py       |                        Bar plot code                         |

## Entorhinal Cortex Reconstruction and Cortical Thickness Calculation

Raw data for the ERC/TEC can be found below. After converting segmentations in meshes, we manually cut out inner and outer surface and runs LDDMM to register the inner surface to the outer surface with the additional constraint that the deformation is restricted to the surface normal direction. We used the lengths of the flow lines, provided by each vertex trajectory under the transformation, to define the vertex-wise thickness between the cortical boundaries. The median of the distribution of the vertex-wise thickness was used to attribute a scalar thickness measurement to each ERC/TEC volume. Surfaces from ADNI are generated from manual segmentation, with only left hemisphere; surfaces from BIOCARD come from nnUNet prediction, with Left and right hemispheres cut and gnerated separately. 

| Dataset |               Directory               |                          Decription                          |
| :-----: | :-----------------------------------: | :----------------------------------------------------------: |
|  ADNI   |   Data/ADNI_Surface/cutted_surfaces   |    Cutted inner and outer surfaces and original surfaces     |
|         |      Data/ADNI_Surface/thickness      |        Calculated thickness of the generated surfaces        |
|         |                                       |                                                              |
| BIOCARD |    Data/BIOCARD_Surface/CUT_MCI_LH    | Cutted inner and outer surfaces and original surfaces, Left Hemisphere |
|         |    Data/BIOCARD_Surface/CUT_MCI_RH    | Cutted inner and outer surfaces and original surfaces, Right Hemisphere |
|         | Data/BIOCARD_Surface/LH_MCI_thickness | Calculated thickness of the generated surfaces, Left Hemisphere |
|         | Data/BIOCARD_Surface/RH_MCI_thickness | Calculated thickness of the generated surfaces, Right Hemisphere |
|         |                                       |                                                              |



Codes for mapping inner surfaces to outer surtfaces and template mapping of population can be found below (Change the input and output direcotry for different datasets):



|                          Directory                           |                          Decription                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|     ThicknessCalculations/runYounesRegistration_ADNI.sh      | run the surface registration by bash ThicknessCalculations/runYounesRegistration_ADNI.sh -s 0 -i Data/ADNI_Surface/cutted_surfaces/ -o Data/ADNI_Surface/mappedsurfaces/ -f all |
| ThicknessCalculations/population_mapping_scripts/AdniMappings.py |           Population template generation for ADNI            |
| ThicknessCalculations/population_mapping_scripts/BiocardNarrowMap.py |      Population template generation for BIOCARD narrow       |
| ThicknessCalculations/population_mapping_scripts/BiocardWideMap.py |       Population template generation for BIOCARD wide        |
|                    Code/F12_TAVA_ADNI.py                     | Figure 12 right panel, plot volume atrophy vs thickness atrophy rate |
|                   Code/F12_TAVA_BIOCARD.py                   | Figure 12 left panel, plot volume atrophy vs thickness atrophy rate |

DEBUGGING required for the population template generation codes； No thickness.vtk files generated for the surface registration code (or at the end have't been generated yet).

## Subregional Amygdalar Atrophy Coupled to High-Field Atlasing

To compute volumetric atrophy within amygdala subregions, we employed a two-stage diffeomorphic mapping procedure from the high-field template to each subject’s 3T segmentation. For each subject, all available time points were jointly aligned in a global LDDMM space using particle based varifold LDDMM from the amygdala, ERC, TEC, and hippocampus. A subject-specific midpoint template was then generated by computing the average initial momentum across time points and deforming the high field atlas. This subject-specific template was subsequently mapped to each individual time point using volume LDDMM.

Following deformation, voxel-wise segmentation masks were reconstructed from the particle representation by interpolating particle-defined anatomical boundaries onto a regular grid and assigning each voxel to the subregion with the highest empirical likelihood. Atrophy rates were computed from the slope of the fitted longitudinal model to the structures.

Code for the subamygdala mapping are summarized below. Follow the steps to get the template substrucutures mapped onto each time point.

| Step |                          Directory                           |                          Decription                          |
| :--: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  1   |          ADNImapping/Codes/makeADNIsegParticles.py           |          Convert ADNI segmentations into particles           |
|  2   |     ADNImapping/Codes/makeHighFieldTemplateParticles.py      | Convert high-field segmentations into particles with 5x downsampling |
|  3   |   ADNImapping/Codes/framework_script_HighFieldToBIOCARD.py   | Stage 1 mapping with three regions (amygdala, ERC/TEC, Hippocampus) from template to all timepoints |
|  4   |           ADNImapping/Codes/templateGeneration.py            | Calculate the average momentum and generate subject-specific template within mapped time points |
|  5   |             ADNImapping/Codes/splitParticles.py              |     Separate the three regions into three particle files     |
|  6   | ADNImapping/Codes/framework_script_HighFieldToBIOCARDIndividual.py | Stage 2 mapping, each region is mapped from the subject-specific template to each time point individually |
|  7   |          ADNImapping/Codes/makeParticlesToImages.py          |    Transform particles into segmentation by interpolation    |
|  8   |             ADNImapping/Codes/VA_mapped_ttest.py             | Calculate the volume atrophy rate of each substructure in amygdala from mapped templates and perform paired t-test |
|  9   |                  Code/F13_bar_VA_subamy.py                   |                    Bar plot for Figure 13                    |

## Tau hemispheric consistency and Tau-VA calculation

 Codes for generating Figure 15 are summarized below:

| Dataset   |             Directory              |                     Decription                     |
| --------- | :--------------------------------: | :------------------------------------------------: |
| ADNI 3, 4 |     Code/F15_ADNI34_tauamy.py      | calculate PET accumulation and volume atrophy rate |
|           | Code/F15_ADNI34_Tauconsistency.py  |          Plot hemishperic tau consistency          |
|           |      Code/F15_ADNI34_TauVA.py      |    Plot tau accumulation vs volume atrophy rate    |
| BIOCARD   |     Code/F15_BIOCARD_tauamy.py     | calculate PET accumulation and volume atrophy rate |
|           | Code/F15_BIOCARD_Tauconsistency.py |          Plot hemishperic tau consistency          |
|           |     Code/F15_BIOCARD_TauVA.py      |    Plot tau accumulation vs volume atrophy rate    |
