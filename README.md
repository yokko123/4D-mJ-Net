# CT Perfusion is All We Need: 4D CNN Segmentation of Penumbra and Core in Patients With Suspected Ischemic Stroke
## Release v1.0
It contains the code described in the paper "CT Perfusion is All We Need: 4D CNN Segmentation of Penumbra and Core in Patients With Suspected Ischemic Stroke".

### 1 - Abstract
Stroke is the second leading cause of death worldwide, and around 87 % of strokes are ischemic strokes.  
Accurate and rapid prediction techniques for identifying ischemic regions, including dead tissue (core) and potentially salvageable tissue (penumbra), in patients with acute ischemic stroke (AIS) hold great clinical importance, as these regions are used to diagnose and plan treatment. 
Computed Tomography Perfusion (CTP) is often used as a primary tool for assessing stroke location, severity, and the volume of ischemic regions.
Current automatic segmentation methods for CTP typically utilize pre-processed 3D parametric maps, traditionally used for clinical interpretation by radiologists. An alternative approach is to use the raw CTP data slice by slice as 2D+time input, where the spatial information over the volume is overlooked. Additionally, these methods primarily focus on segmenting core regions, yet predicting penumbra regions can be crucial for treatment planning.

This paper investigates different methods to utilize the entire raw 4D CTP as input to fully exploit the spatio-temporal information, leading us to propose a 4D convolution layer in a 4D CNN network.
Our comprehensive experiments on a local dataset of 152 patients divided into three groups show that our proposed models generate more precise results than other methods explored.
Adopting the proposed _4D mJ-Net_, a Dice Coefficient of 0.53 and 0.23 is achieved for segmenting penumbra and core areas, respectively.
Using the entire 4D CTP data for AIS segmentation offers improved precision and potentially better treatment planning in patients suspected of this condition.

### 2 - Link to paper

https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10328592

### 3 - Dependecies:
```
conda env create -f environment.yml
conda activate mj-net
```
### 4 - Preparing the Dataset
To prepare the dataset, first ensure that the dataset and the ground truths are `registered`, parametric maps are ready, brain masks are ready. Make sure you know the correct dimension of the images and correct number of timepoints. You also need to ensure that the labels and the label thresholds are correct. For preparing the dataset, we need to run the (get_complete_training_data.py)[https://github.com/yokko123/4D-mJ-Net/blob/main/Standalone-scripts/get_complete_training_data.py] script. The script will generate pickle files for each patient. Each pickle file will contain these information ["patient_id", "label", "pixels", "CBF", "CBV", "TTP", "TMAX", "MIP", "NIHSS", "ground_truth", "mask", "label_code", "x_y", "data_aug_idx", "timeIndex", "sliceIndex", "severity", "age", "gender"]. 
#### Folder Structure of the Dataset
```
CTP_01_001/
├── 01
├── 02
├── 03
├── 04
├── 05
├── 06
├── 07
├── 08
├── 09
├── 10
├── 11
├── 12
├── 13
└── 14

```
The folders are structured slice by slice. Inside this folders there are 30 timepoint .tiff images. 
```
CTP_01_001/01
├── 01.tiff
├── 02.tiff
├── 03.tiff
├── 04.tiff
├── 05.tiff
├── 06.tiff
├── 07.tiff
├── 08.tiff
├── 09.tiff
├── 10.tiff
├── 11.tiff
├── 12.tiff
├── 13.tiff
├── 14.tiff
├── 15.tiff
├── 16.tiff
├── 17.tiff
├── 18.tiff
├── 19.tiff
├── 20.tiff
├── 21.tiff
├── 22.tiff
├── 23.tiff
├── 24.tiff
├── 25.tiff
├── 26.tiff
├── 27.tiff
├── 28.tiff
├── 29.tiff
└── 30.tiff
```
To run the script, we need a configuration file. 
```json
{
  "DATASET_NAME": "",
  "ROOT_PATH": "/home/prosjekt/IschemicStroke/",
  "SCRIPT_PATH": "/home/stud/sazidur/bhome/sus2020_training/",
  "SAVE_REGISTERED_FOLDER": "/home/prosjekt/IschemicStroke/Data/CTP/Pre-processed/SUS2020/Dataset/IMAGES_1_0_1_0.5/",
  "PM_FOLDER": "/home/prosjekt/IschemicStroke/Data/CTP/Pre-processed/SUS2020/Dataset/Parametric_Maps/",
  "LABELED_IMAGES_FOLDER_LOCATION": "/home/prosjekt/IschemicStroke/Data/CTP/Ground_Truth/SUS2020/Dataset/",
  "MASKS_IMAGES_FOLDER_LOCATION": "/home/prosjekt/IschemicStroke/Data/CTP/Pre-processed/SUS2020/Dataset/Masks/",
  "IMAGE_PREFIX": "CTP_",
  "GT_SUFFIX": ".tiff",
  "IMAGE_SUFFIX": ".tiff",
  "NUMBER_OF_IMAGE_PER_SECTION": 30,
  "IMAGE_WIDTH": 512,
  "IMAGE_HEIGHT": 512,
  "BINARY_CLASSIFICATION": 1,
  "_LABELS": ["background", "core"],
  "_LABELS_THRESHOLDS": [0, 230],
  "_LABELS_REALVALUES": [0, 255],
  "LABELS": ["background", "brain", "penumbra", "core"],
  "LABELS_THRESHOLDS": [0, 70, 155, 230],
  "LABELS_REALVALUES": [0, 85, 170, 255],
  "NEW_GROUNDTRUTH_VALUES": 1,
  "TILE_DIVISION": 1,
  "SEQUENCE_DATASET": 1,
  "SKIP_TILES": 0,
  "ORIGINAL_SHAPE": 0,
  "DATA_AUGMENTATION": 1,
  "ONE_TIME_POINT": -1,
  "COLUMNS": ["patient_id", "label", "pixels", "CBF", "CBV", "TTP", "TMAX", "MIP", "NIHSS", "ground_truth", "mask",
           "label_code", "x_y", "data_aug_idx", "timeIndex", "sliceIndex", "severity", "age", "gender"]
}
```
Change the paths according to your dataset.

## ✅ To Do
- [x] Upgrading code for new packages and python version
- [x] train on new conda environment
- [ ] Write documentation 📚  

### How to cite our work
The code is released free of charge as open-source software under the GPL-3.0 License. Please cite our paper when you have used it in your study.
```
TBA
```

### Got Questions?
Email the author at luca.tomasetti@uis.no

