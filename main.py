#!/usr/bin/python

import sys

from Utils import general_utils, dataset_utils
import constants
from NeuralNetworkClass import NeuralNetwork

# to remove *SOME OF* the warning from tensorflow (regarding deprecation) <-- to remove when update tensorflow!
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# MAIN FUNCTION
def main():
    networks = dict()
    train_df = None

    # Get the command line arguments
    args = general_utils.getCommandLineArguments()

    # Read the settings from json file
    setting = general_utils.getSettingFile(args.sname)

    # set up the environment for GPUs
    n_gpu = general_utils.setupEnvironment(args, setting)

    # initialize model(s)
    for name, info in setting["models"].items():
        networks[name] = NeuralNetwork(info, setting)

    for key, nn in networks.items():
        for testPatient in setting["PATIENT_TO_TEST"]:
            p_id = general_utils.getStringPatientIndex(testPatient)
            isAlreadySaved = False

            # set the multi/single PROCESSING
            nn.setProcessingEnv(setting["init"]["MULTIPROCESSING"])
            # Training or Test ?

            # Check if the model was already trained and saved
            if nn.isModelSaved(p_id):
                nn.loadSavedModel(p_id)
                isAlreadySaved = True
            else:
                ## GET THE DATASET
                # if we are using a data augmentation dataset we need to get the dataset differently each time
                if nn.da: train_df = dataset_utils.getDataset(nn, p_id)
                else: # Otherwise get dataset only the first time
                    if train_df is None: train_df = dataset_utils.getDataset(nn)

                ## PREPARE DATASET
                nn.prepareDataset(train_df, p_id)
                ## SET THE CALLBACKS, RUN TRAINING & SAVE THE MODELS WEIGHTS
                nn.setCallbacks(p_id)
                nn.runTraining(p_id, n_gpu)
                nn.saveModelAndWeight(p_id)

            ## PERFOM TESTING
            if nn.supervised:
                nn.evaluateModelWithCategorics(p_id, isAlreadySaved)
            # predict and save the images
            nn.predictAndSaveImages(p_id)






if __name__ == '__main__':
    main()
