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
To prepare the dataset, first ensure that the dataset and the ground truths are `registered`, parametric maps are ready, brain masks are ready. Make sure you know the correct dimension of the images and correct number of timepoints. You also need to ensure that the labels and the label thresholds are correct. 

For preparing the dataset, we need to run the [get_complete_training_data.py](https://github.com/yokko123/4D-mJ-Net/blob/main/Standalone-scripts/get_complete_training_data.py) script. The script will generate pickle files for each patient. Each pickle file will contain these information `["patient_id", "label", "pixels", "CBF", "CBV", "TTP", "TMAX", "MIP", "NIHSS", "ground_truth", "mask", "label_code", "x_y", "data_aug_idx", "timeIndex", "sliceIndex", "severity", "age", "gender"]`. 
#### Folder Structure of the Dataset
```
CTP_01_001/
├── 01/
│   ├── 01.tiff
│   ├── 02.tiff
│   └── ...
│   └── 30.tiff
├── 02/
├── ...
└── 14/
```
To run the script, we need a `configuration` file. 
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
⚠️ Update all paths to reflect your local system setup. The dataset needs to be in the exact folder structure. Otherwise the script won't work. 
#### Run the Script
```sh
python Standalone-scripts/get_complete_training_data.py SUS2020.json --verbose

```
This will save the Pickle files in the designated folder. The structure will be like this:
```
sus2020_training/
├── patient00_007_128_512x512.pkl
├── patient00_009_128_512x512.pkl
├── patient01_001_128_512x512.pkl
├── patient01_002_128_512x512.pkl
├── patient01_003_128_512x512.pkl
├── patient01_004_128_512x512.pkl
├── patient01_005_128_512x512.pkl
......
....
....
├── patient03_010_128_512x512.pkl
├── patient03_011_128_512x512.pkl
├── patient03_012_128_512x512.pkl
├── patient03_013_128_512x512.pkl
├── patient03_014_128_512x512.pkl
└── patient03_015_128_512x512.pkl
```
### 5 - Training the 4DmJ-Net
Now that you have the dataset in proper format, you can now run the `4DmJ-Net`. But before training, you will need a configuration file just like you needed one while generating the training ready data.
```json

  "EXPERIMENT": 10, %% # of experiment
  "root_path": "/home/stud/lucat/PhD_Project/Stroke_segmentation/", %% main path
  "dataset_path": "/local/home/lucat/DATASET/ISLES2018/", % dataset path
   %% list of patients to test (if == ["ALL"] test all of them)
  "PATIENTS_TO_TRAINVAL": ["ALL"],
  "PATIENTS_TO_TEST": ["..."], %% list of patient to exclude
  "OVERRIDE_MODELS_ID_PATH": "", %% path for overriding the model ID
  "init": { %% basic flag for init the process
    "NUMBER_OF_IMAGE_PER_SECTION": 64, %% change the number of image per section (optional)
    "3D": 1, %% flag for the 3D dataset (optional)
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "per_process_gpu_memory_fraction": 0.5,
    "allow_growth": 1,
    "MULTIPROCESSING": 0
  },
  %% paths containing the various important folders: save, labeled_images, ...
  "relative_paths": {
    "labeled_images": "PATIENTS/ISLES2018/NEW_TRAINING/Ground Truth/",
    "patients": "PATIENTS/ISLES2018/NEW_TRAINING/FINAL/",
    "save": {
      "model": "MODELS/",
      "partial_model": "TMP_MODELS/",
      "plot": "PLOTS/",
      "images": "IMAGES/",
      "text": "TEXT/",
      "intermediate_activation": "TMP/" %% (optional) folder for the intermediate activation images
    }
  },
  %% definition of the model(s)
  "models": [
     {
      "name": "mJNet_v2", %% name of nn, same as the name function
      "loss": "mod_dice_coef_loss", %% loss name function
      "metrics": ["mod_dice_coef", "dice_coef"], %% list of metric name functions
      "epochs": 50, %% # of epochs
      "batch_size":8, %% set batch size (optional, default=32)
      %% validation variable
      "val":{
        "validation_perc": 5, %% percentage
        "number_patients_for_validation": 5, % number of patients for the validation set
        "number_patients_for_testing": 4, % number of patients for the test set
        "random_validation_selection": 0 % flag for random selection in the validation dataset
      },
      "test_steps":1, %% # of test steps
      %% optimizer info (different for each of them (ADAM, SGD, ...))
      "optimizer": {
        "name": "SGD",
        "learning_rate": 0.01,
        "decay": 1e-6,
        "momentum":0.9,
        "nesterov":"True"
      },
      %% choseable parameters for the nn
      "params":{
        "dropout":{ %% dropout value for layers
          "0.1":0.25,
          "1":0.25,
          "2":0.25,
          "3":0.25,
          "3.1":0.5,
          "3.2":0.5,
          "3.3":0.5,
          "3.4":0.5,
          "4":0.3,
          "5":0.3
        },
        "max_pool":{ %% max pooling values for the longJ layers
          "long.1":3,
          "long.2":2,
          "long.3":5
        }
      },
      %% list of callbacks with choseable parameters
      "callbacks":{
        "ModelCheckpoint": { %% save the model
          "monitor": "mod_dice_coef",
          "mode": "max",
          "period": 1
        },
        "EarlyStopping": { %% stop the training based on parameters
          "monitor": "loss",
          "min_delta": 0.001,
          "patience": 12
        },
        "ReduceLROnPlateau": { %% reduce learning rate
          "monitor": "val_loss",
          "factor": 0.1,
          "patience": 2,
          "min_delta": 1e-4,
          "cooldown": 1,
          "min_lr": 0
        },
        "CollectBatchStats": { %% save stats
          "acc":"mod_dice_coef"
        }
      },
      %% important flags (1=yes, 0=no)
      "to_categ":0, %% select categorical output in the nn (softmax activation)
      "save_images": 1, %% save the images
      "data_augmentation": 1, %% use data augmention dataset
      "cross_validation": 0, %% use cross validation during training (save various models)
      "train_again": 0, %% train again even if the model is already saved
      "supervised": 1, %% supervised learning?
      "save_activation_filter": 0, %% save the activation filter at the end of the training
      "use_hickle": 1 %% flag to load hickle dataset instead of pickle
    }
  ]
}
```
Once your settings/config file is ready, run the training script.

For the training arguments please check:
```sh
    Usage: python main.py gpu sname
                [-h] [-v] [-d] [-o] [-pm] [-t TILE] [-dim DIMENSION] [-c {2,3,4}] [-w ...] [-e EXP] [-j]  [--timelast] [--prefix PREFIX]

    positional arguments:
      gpu                   Give the id of gpu (or a list of the gpus) to use
      sname                 Select the setting filename

    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         Increase output verbosity
      -d, --debug           DEBUG mode
      -o, --original        Set the shape of the testing dataset to be compatible with the original shape 
      -pm, --pm             Set the flag to train the parametric maps as input 
      -t TILE, --tile TILE  Set the tile pixels dimension (MxM) (default = 512)
      -dim DIMENSION, --dimension DIMENSION
                            Set the dimension of the input images (width X height) (default = 512)
      -c {2,3,4}, --classes {2,3,4}
                            Set the # of classes involved (default = 3)
      --isles2018           Flag to use the ISLES2018 dataset
      -w, --weights         Set the weights for the categorical losses
      -e, --exp             Set the number of the experiment
      -j, --jump            Jump the training and go directly on the gradual fine-tuning function
      --test                Flag for predicting the test patients 
      --timelast            Set the time dimension in the last channel of the input model          
      --prefix              Set the prefix different from the default
      --limcols             Set the columns without additional info 
      --sweep               Flag to set the sweep for WandB 
      --array               Flag for setting the sbatch array modality*
```
The model weights,test outputs and plots will be saved in `SAVE` folder. 
### 6 - Inference
The inference script is under works currently. 
If you want to check you can run

```sh
python predict.py
```
But make sure to change 
```python
CONFIG_PATH = "/home/stud/sazidur/bhome/4D-mJ-Net/SAVE/EXP036.3/setting.json"
WEIGHTS_PATH = "/home/stud/sazidur/bhome/4D-mJ-Net/SAVE/EXP036.3/TMP_MODELS/mJNet_3dot5D_DA_ADAM_VAL20_SOFTMAX_128_512x512__69.h5"
INPUT_FOLDER = "/home/stud/sazidur/bhome/sus-nifti/0001/baseline_tiff/7/"
OUTPUT_FOLDER = "/home/stud/sazidur/bhome/preprocess_isles_amador_512_5mm/output/0001/baseline/7/"

```
Update these things accordingly. 

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

