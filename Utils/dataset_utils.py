import warnings
import glob, random, time
import multiprocessing
import pickle as pkl
import hickle as hkl
import numpy as np
import pandas as pd
import sklearn
from tensorflow.keras import utils

from Model.constants import *
from Utils import general_utils

warnings.simplefilter(action='ignore', category=FutureWarning)


################################################################################
# Function to load the saved dataframes based on the list of patients
def load_train_DF(nn, patients):
    train_df = pd.DataFrame(columns=get_DF_columns())
    # get the suffix based on the SLICING_PIXELS, the M and N
    suffix = general_utils.get_suffix()  # es == "_4_16x16"
    suffix_filename = ".pkl"
    if nn.model_info["use_hickle"]: suffix_filename = ".hkl"
    listOfFolders = glob.glob(nn.ds_folder + "*" + suffix + suffix_filename)
    # print("[INFO]", listOfFolders)
    with multiprocessing.Pool(processes=5) as pool:  # auto closing workers
        frames = pool.starmap(read_single_DF, list(zip(listOfFolders, [patients] * len(listOfFolders),
                                                       [suffix] * len(listOfFolders), [get_m()] * len(listOfFolders),
                                                       [get_n()]*len(listOfFolders), [get_labels()]*len(listOfFolders),
                                                       [is_verbose()] * len(listOfFolders), [nn.model_info["use_hickle"]] * len(listOfFolders))))
        print("PATIENTS",patients)
    if is_ISLES2018(): train_df = train_df.append(frames[1:], sort=False, ignore_index=True)
    else: train_df = train_df.append(frames, sort=False, ignore_index=True)
    return train_df


################################################################################
# Useful function to lead a single pandas DataFrame
def read_single_DF(filename_train, patients, suffix, thisM, thisN, thisLABELS, thisVerbose, use_hickle):
    # don't load the dataframe if patient_id NOT in the list of patients
    tmp_df = pd.DataFrame(columns=get_DF_columns())
    if not general_utils.is_filename_in_patientlist(filename_train, patients, suffix): return tmp_df
    s = time.time()
    tmp_df = read_pickle_or_hickle(filename_train, use_hickle)
    # Remove the overlapping tiles except if they are labeled as "core"
    one = tmp_df.x_y.str[0] % thisM == 0
    two = tmp_df.x_y.str[1] % thisN == 0
    three = tmp_df.label.values == thisLABELS[-1]
    tmp_df = tmp_df[(one & two) | three]

    # if thisVerbose: print("{0} - {2} - {1}".format(filename_train, round(time.time() - s, 3), tmp_df.shape))
    return tmp_df


################################################################################
# Return the elements in the filename saved as a pickle or as hickle (depending on the flag)
def read_pickle_or_hickle(filename, flagHickle):
    if flagHickle: return sklearn.utils.shuffle(hkl.load(filename))
    else:
        file = open(filename, "rb")
        return sklearn.utils.shuffle(pkl.load(file))


################################################################################
################################################################################
# Return the dataset based on the patient id
# First function that is been called to create the train_df!
def get_ds(nn, listOfPatientsToTrainVal):
    if is_verbose():
        general_utils.print_sep("-", 50)
        if nn.mp: print("[INFO] - Loading Dataset using MULTI-processing...")
        else: print("[INFO] - Loading Dataset using SINGLE-processing...")

    train_df = load_train_DF(nn, patients=listOfPatientsToTrainVal)
    if is_verbose(): generate_ds_summary(train_df, listOfPatientsToTrainVal)  # summary of the dataset

    return train_df


################################################################################
# Function to divide the dataframe in train and test based on the patient id;
# plus it reshape the pixel array and initialize the model.
def split_ds(nn, patientlist_train_val, patientlist_test):
    validation_list, test_list = list(), list()

    for flag in ["train", "val", "test"]: nn.dataset[flag]["indices"] = list()
    if nn.cross_validation["use"]==0:  # here only if: NO cross-validation set
        # We have set a number of testing patient(s) and we are inside a supervised learning
        if nn.model_info["supervised"]:
            if nn.val["number_patients_for_testing"] > 0 or len(patientlist_test) > 0:
                random.seed(nn.val["seed"])
                # if we already set the patient list in the setting file
                if len(patientlist_test) > 0: test_list = patientlist_test
                else: test_list = random.sample(patientlist_train_val, nn.val["number_patients_for_testing"])
                # remove the test_list elements from the list
                patientlist_train_val = list(set(patientlist_train_val).difference(test_list))
                if is_verbose(): print("[INFO] - TEST list: {}".format(test_list))

                for test_p in test_list: nn.dataset["test"]["indices"].extend(np.nonzero((nn.train_df.patient_id.values == general_utils.get_str_from_idx(
                    test_p)))[0])
        # We have set a number of validation patient(s)
        if nn.val["number_patients_for_validation"] > 0:
            patientlist_train_val.sort(reverse=False)  # sort the list and then...
            random.seed(nn.val["seed"])
            random.shuffle(patientlist_train_val)  # shuffle it
            validation_list = random.sample(patientlist_train_val, nn.val["number_patients_for_validation"])
            nn = set_val_list(nn, validation_list)

        nn = set_train_indices(nn, validation_list, test_list)
    else:  # We are doing a cross-validation!
        # Select the list of validation patients based on the split
        if is_verbose(): print("[INFO] - VALIDATION SPLIT #: {}".format(nn.model_split))
        # We assume that the TEST list is already set
        patientlist_train_val.sort(reverse=False)  # sort the list and then...
        random.seed(nn.val["seed"])
        random.shuffle(patientlist_train_val)  # shuffle it
        n_val_pat = int(np.ceil(len(patientlist_train_val) / nn.cross_validation["split"]))
        validation_list = patientlist_train_val[(int(nn.model_split) - 1) * n_val_pat:int(nn.model_split) * n_val_pat]
        nn = set_val_list(nn, validation_list)
        nn = set_train_indices(nn, validation_list, test_list)
    return nn.dataset, validation_list, test_list


################################################################################
# Print info regarding the validation list and set the indices in the nn dataset
def set_val_list(nn, validation_list):
    if is_verbose():
        if get_prefix_img() == "CTP_":
            print("[INFO] - VALIDATION list LVO: {}".format([v for v in validation_list if "01_" in v or "00_" in v or "21_" in v or "20_" in v]))
            print("[INFO] - VALIDATION list Non-LVO: {}".format([v for v in validation_list if "02_" in v or "22_" in v]))
            print("[INFO] - VALIDATION list WIS: {}".format([v for v in validation_list if "03_" in v or "23_" in v]))
        else: print("[INFO] - VALIDATION list {}".format(validation_list))
    for val_p in validation_list: nn.dataset["val"]["indices"].extend(np.nonzero((nn.train_df.patient_id.values == general_utils.get_str_from_idx(val_p)))[0])

    return nn


################################################################################
# Set the TRAIN indices in the nn dataset
def set_train_indices(nn, validation_list, test_list):
    # Set the indices for the train dataset as the difference between all_indices,
    # the validation indices and the test indices
    all_indices = np.nonzero((nn.train_df.label_code.values >= 0))[0]
    nn.dataset["train"]["indices"] = list(set(all_indices) - set(nn.dataset["val"]["indices"]) - set(nn.dataset["test"]["indices"]))

    # the validation list is empty, the test list contains some patient(s) and the validation_perc > 0
    if len(validation_list) == 0 and len(test_list) > 0 and nn.val["validation_perc"] > 0:
        train_val_dataset = nn.dataset["train"]["indices"]
        nn.dataset["train"]["indices"] = list()  # empty the indices, it will be set inside the next function
        nn = get_rnd_or_weighted_val_selection(nn, train_val_dataset)

    return nn


################################################################################
# Prepare the dataset (NOT for the sequence class!!)
def prepare_ds(nn):
    # set the train data
    nn.dataset["train"]["data"] = get_data_from_idx(nn.train_df, nn.dataset["train"]["indices"], "train", nn.mp)
    # the validation data is None if validation_perc and number_patients_for_validation are BOTH equal to 0
    nn.dataset["val"]["data"] = None if nn.val["validation_perc"] == 0 and nn.val["number_patients_for_validation"] == 0 else get_data_from_idx(
        nn.train_df, nn.dataset["val"]["indices"], "val", nn.mp)

    # DEFINE the data for the dataset TEST
    nn.dataset["test"]["data"] = get_data_from_idx(nn.train_df, nn.dataset["test"]["indices"], "test", nn.mp)

    return nn.dataset


################################################################################
# Return the train and val indices based on a random selection (val_mod) if nn.val["random_validation_selection"]
# or a weighted selection based on the percentage (nn.val["validation_perc"]) for each class label
def get_rnd_or_weighted_val_selection(nn, train_val_dataset):
    # perform a random selection of the validation
    if nn.val["random_validation_selection"]:
        val_mod = int(100 / nn.val["validation_perc"])
        nn.dataset["train"]["indices"] = np.nonzero((train_val_dataset % val_mod != 0))[0]
        nn.dataset["val"]["indices"] = np.nonzero((train_val_dataset % val_mod == 0))[0]
    else:
        # do NOT use a patient(s) as a validation set because maybe it doesn't have
        # too much information about core and/or penumbra. Instead, get a percentage from each class!
        for classLabelName in get_labels():
            random.seed(0)  # use ALWAYS the same random indices
            classIndices = np.nonzero((nn.train_df.label.values[train_val_dataset] == classLabelName))[0]
            classValIndices = [] if nn.val["validation_perc"] == 0 else random.sample(list(classIndices), int(
                (len(classIndices) * nn.val["validation_perc"]) / 100))
            nn.dataset["train"]["indices"].extend(list(set(classIndices) - set(classValIndices)))
            if nn.val["validation_perc"] > 0: nn.dataset["val"]["indices"].extend(classValIndices)

    return nn


################################################################################
# Get the test dataset, where the test indices are == p_id
def get_test_ds(dataset, train_df, p_id, use_sequence, mp):
    dataset["test"]["indices"] = np.nonzero((train_df.patient_id.values == p_id))[0]
    if not use_sequence: dataset["test"]["data"] = get_data_from_idx(train_df, dataset["test"]["indices"], "test", mp)
    return dataset


################################################################################
# Get the data from a list of indices
def get_data_from_idx(train_df, indices, flag, mp):
    start = time.time()
    if is_3D() != "":
        data = [a for a in np.array(train_df.pixels.values[indices], dtype=object)]
    else:  # do this when NO 3D flag is set
        # reshape the data adding a last (1,)
        data = [a.reshape(a.shape + (1,)) for a in np.array(train_df.pixels.values[indices], dtype=object)]

    # convert the data into an np.ndarray
    if type(data) is not np.ndarray: data = np.array(data, dtype=object)

    if is_verbose():
        setPatients = set(train_df.patient_id.values[indices])
        print("[INFO] - patients: {0}".format(setPatients))
        print("[INFO] - *getDataFromIndex* Time: {}s".format(round(time.time() - start, 3)))
        print("[INFO] - {0} shape; # {1}".format(data.shape, flag))

    return data


################################################################################
# Function that reshape the data in a MxN tile
def get_single_label_from_idx(singledata): return singledata.reshape(get_m(), get_n())


################################################################################
# Function that convert the data into a categorical array based on the number of classes
def get_single_label_from_idx_categ(singledata,n_classes):
    return np.array(utils.to_categorical(np.rint((singledata/255) * (n_classes - 1)), num_classes=n_classes))


################################################################################
# Return the labels given the indices
def get_labels_from_idx(train_df, dataset, modelname, flag):
    start = time.time()
    labels = None
    indices = dataset["indices"]

    # if we are using an autoencoder, the labels are the same as the data!
    if modelname.find("autoencoder") > -1: return dataset["data"]

    data = [a for a in np.array(train_df.ground_truth.values[indices])]

    if is_TO_CATEG():
        with multiprocessing.Pool(processes=1) as pool:  # auto closing workers
            labels = pool.map(get_single_label_from_idx_categ, list(zip(data, [get_n_classes()] * len(data))))
        if type(labels) is not np.array: labels = np.array(labels)
    else:
        if get_n_classes() == 3:
            for i, curr_data in enumerate(data):
                data[i][curr_data == 85] = get_pixel_values()[0]  # remove one class from the ground truth
                #data[i][curr_data == 150] = getPIXELVALUES()[2]  # change the class for core
        if type(data) is not np.array: data = np.array(data)
        labels = data.astype(np.float32)
        labels /= 255  # convert the label in [0, 1] values

    if is_verbose():
        print("[INFO] - *getLabelsFromIndex* Time: {}s".format(round(time.time() - start, 3)))
        print("[INFO] - {0} shape; # {1}".format(labels.shape, flag))

    return labels


################################################################################
# Generate a summary of the dataset
def generate_ds_summary(train_df, listOfPatientsToTrainVal=None):
    N_BACKGROUND, N_BRAIN, N_PENUMBRA, N_CORE, N_TOT, back_perc = get_number_of_elem(train_df)

    general_utils.print_sep('+', 100)
    print("DATASET SUMMARY: \n")
    print("\t N. Background: {0}".format(N_BACKGROUND))
    if get_n_classes() >3: print("\t N. Brain: {0}".format(N_BRAIN))
    if get_n_classes() >2: print("\t N. Penumbra: {0}".format(N_PENUMBRA))
    print("\t N. Core: {0}".format(N_CORE))
    print("\t Rest/Background (%): {}".format(back_perc))
    print("\t Tot: {0}".format(N_TOT))

    if listOfPatientsToTrainVal is not None: print("\t Patients: {0}".format(listOfPatientsToTrainVal))
    general_utils.print_sep('+', 100)


################################################################################
# Return the number of element per class of the dataset
def get_number_of_elem(train_df):
    N_BRAIN, N_PENUMBRA, back_perc = 0, 0, 0
    back_v, brain_v, penumbra_v, core_v = get_labels()[0], "brain", "penumbra", get_labels()[-1]
    N_BACKGROUND = len([x for x in train_df.label if x == back_v])
    N_CORE = len([x for x in train_df.label if x == core_v])
    N_BRAIN = len([x for x in train_df.label if x == brain_v])
    if get_n_classes()<=3: N_BACKGROUND+=N_BRAIN
    if get_n_classes()>2: N_PENUMBRA = len([x for x in train_df.label if x == penumbra_v])
    N_TOT = train_df.shape[0]
    if get_n_classes()==2: back_perc = N_CORE/N_BACKGROUND
    elif get_n_classes()==3: back_perc = (N_CORE+N_PENUMBRA)/N_BACKGROUND
    else: back_perc = (N_CORE+N_PENUMBRA+N_BRAIN)/N_BACKGROUND
    back_perc*=100
    return N_BACKGROUND, N_BRAIN, N_PENUMBRA, N_CORE, N_TOT, round(back_perc,3)
