from Utils import general_utils, dataset_utils, sequence_utils, models
import training, testing, constants

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import multi_gpu_model, plot_model


################################################################################
# Class that defines a NeuralNetwork
################################################################################
class NeuralNetwork(object):
    """docstring for NeuralNetwork."""

    def __init__(self, modelInfo, setting):
        super(NeuralNetwork, self).__init__()
        self.summaryFlag = 0

        # Used to override the path for the saved model in order to test patients with a specific model
        self.OVERRIDE_MODELS_ID_PATH = setting["OVERRIDE_MODELS_ID_PATH"] if setting["OVERRIDE_MODELS_ID_PATH"]!="" else False

        self.name = modelInfo["name"]
        self.epochs = modelInfo["epochs"]
        self.train_on_batch = modelInfo["train_on_batch"] if "train_on_batch" in modelInfo.keys() else 0
        self.batch_size = modelInfo["batch_size"] if "batch_size" in modelInfo.keys() else 32

        self.val = {
            "validation_perc": modelInfo["val"]["validation_perc"],
            "random_validation_selection": modelInfo["val"]["random_validation_selection"],
            "number_patients_for_validation": modelInfo["val"]["number_patients_for_validation"] if "number_patients_for_validation" in modelInfo["val"].keys() else 0,
            "number_patients_for_testing": modelInfo["val"]["number_patients_for_testing"] if "number_patients_for_testing" in modelInfo["val"].keys() else 0
        }
        self.test_steps = modelInfo["test_steps"]

        self.dataset = {
            "train": {},
            "val": {},
            "test": {}
        }

        # get parameter for the model
        self.optimizerInfo = modelInfo["optimizer"]
        self.params = modelInfo["params"]
        self.loss = general_utils.getLoss(modelInfo["loss"])
        self.classes_to_evaluate = modelInfo["classes_to_evaluate"]
        self.metricFuncs = general_utils.getStatisticFunctions(modelInfo["metrics"])
        self.statistics = general_utils.getStatisticFunctions(modelInfo["statistics"])

        # FLAGS for the model
        self.to_categ = True if modelInfo["to_categ"]==1 else False
        self.save_images = True if modelInfo["save_images"]==1 else False
        self.save_statistics = True if modelInfo["save_statistics"]==1 else False
        self.use_background_in_statistics = True if modelInfo["use_background_in_statistics"]==1 else False
        self.calculate_ROC = True if modelInfo["calculate_ROC"]==1 else False
        self.da = True if modelInfo["data_augmentation"]==1 else False
        self.train_again = True if modelInfo["train_again"]==1 else False
        self.cross_validation = True if modelInfo["cross_validation"]==1 else False
        self.supervised = True if modelInfo["supervised"]==1 else False
        self.save_activation_filter = True if modelInfo["save_activation_filter"]==1 else False
        self.use_hickle = True if "use_hickle" in modelInfo.keys() and modelInfo["use_hickle"]==1 else False

        # paths
        self.rootPath = setting["root_path"]
        self.datasetFolder = setting["dataset_path"]
        self.labeledImagesFolder = setting["relative_paths"]["labeled_images"]
        self.patientsFolder = setting["relative_paths"]["patients"]
        self.experimentID = "EXP"+general_utils.convertExperimentNumberToString(setting["EXPERIMENT"])
        self.experimentFolder = "SAVE/" + self.experimentID + "/"
        self.savedModelFolder = self.experimentFolder+setting["relative_paths"]["save"]["model"]
        self.savePartialModelFolder = self.experimentFolder+setting["relative_paths"]["save"]["partial_model"]
        self.saveImagesFolder = self.experimentFolder+setting["relative_paths"]["save"]["images"]
        self.savePlotFolder = self.experimentFolder+setting["relative_paths"]["save"]["plot"]
        self.saveTextFolder = self.experimentFolder+setting["relative_paths"]["save"]["text"]
        self.intermediateActivationFolder = self.experimentFolder+setting["relative_paths"]["save"]["intermediate_activation"] if "intermediate_activation" in setting["relative_paths"]["save"].keys() else None

        self.infoCallbacks = modelInfo["callbacks"]
        self.model = None

        if constants.N_CLASSES == 4: self.epsilons = [(63.75-constants.PIXELVALUES[1]-1, 127.5-constants.PIXELVALUES[2], 191.25-constants.PIXELVALUES[3])]
        else: self.epsilons = [(63.75-constants.PIXELVALUES[1]-1)]

        # epsiloList is the same list of epsilons multiply for the percentage (thresholding) involved to calculate ROC
        self.epsiloList = list(range(0,110, 10)) if self.calculate_ROC else [None]

        # change the prefix if SUS2020_v2 is in the dataset name
        if "SUS2020" in self.datasetFolder: constants.setPrefixImagesSUS2020_v2()

################################################################################
# Initialize the callbacks
    def setCallbacks(self, p_id, sample_weights=None):
        if self.getVerbose():
            general_utils.printSeparation("-", 50)
            print("[INFO] - Setting callbacks...")

        self.callbacks = training.getCallbacks(
            root_path=self.rootPath,
            info=self.infoCallbacks,
            filename=self.getSavedInformation(p_id, path=self.savePartialModelFolder),
            textFolderPath=self.saveTextFolder,
            dataset=self.dataset,
            sample_weights=sample_weights # only for ROC callback (NOT working)
        )

################################################################################
# return a Boolean to control if the model was already saved
    def isModelSaved(self, p_id):
        saved_modelname = self.getSavedModel(p_id)
        saved_weightname = self.getSavedWeight(p_id)

        return os.path.isfile(saved_modelname) and os.path.isfile(saved_weightname)

################################################################################
# load json and create model
    def loadSavedModel(self, p_id):
        saved_modelname = self.getSavedModel(p_id)
        saved_weightname = self.getSavedWeight(p_id)
        json_file = open(saved_modelname, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(saved_weightname)

        if self.getVerbose():
            general_utils.printSeparation("+",100)
            print("[INFO - Loading] - --- MODEL {} LOADED FROM DISK! --- ".format(saved_modelname))
            print("[INFO - Loading] - --- WEIGHTS {} LOADED FROM DISK! --- ".format(saved_weightname))

################################################################################
# Check if there are saved partial weights
    def arePartialWeightsSaved(self, p_id):
        self.partialWeightsPath = ""
        # path ==> weight name plus a suffix ":" <-- constants.suffix_partial_weights
        path = self.getSavedInformation(p_id, path=self.savePartialModelFolder)+constants.suffix_partial_weights
        for file in glob.glob(self.savePartialModelFolder+"*.h5"):
            if path in self.rootPath+file: # we have a match
                self.partialWeightsPath = file
                return True

        return False

################################################################################
# Load the partial weights and set the initial epoch where the weights were saved
    def loadModelFromPartialWeights(self, p_id):
        if self.partialWeightsPath!="":
            self.model.load_weights(self.partialWeightsPath)
            self.initial_epoch = general_utils.getEpochFromPartialWeightFilename(self.partialWeightsPath)

            if self.getVerbose():
                general_utils.printSeparation("+",100)
                print("[INFO - Loading] - --- WEIGHTS {} LOADED FROM DISK! --- ".format(self.partialWeightsPath))
                print("[INFO] - --- Start training from epoch {} --- ".format(str(self.initial_epoch)))

################################################################################
# Function to divide the dataframe in train and test based on the patient id;
    def splitDataset(self, train_df, p_id, listOfPatientsToTrainVal, listOfPatientsToTest):
        # set the dataset inside the class
        self.train_df = train_df
        # split the dataset
        self.dataset, self.test_list = dataset_utils.splitDataset(self, p_id, listOfPatientsToTrainVal, listOfPatientsToTest)
        # get the number of element per class in the dataset
        self.N_BACKGROUND, self.N_BRAIN, self.N_PENUMBRA, self.N_CORE, self.N_TOT = dataset_utils.getNumberOfElements(self.train_df)

################################################################################
#Function to reshape the pixel array and initialize the model.
    def prepareDataset():
        # split the dataset
        self.dataset = dataset_utils.prepareDataset(self)

################################################################################
# compile the model, callable also from outside
    def compileModel(self):
        # set the optimizer (or reset)
        self.optimizer = training.getOptimizer(optInfo=self.optimizerInfo)

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss["loss"],
            metrics=[self.metricFuncs]
        )

################################################################################
#
    def initializeTraining(self, p_id, n_gpu):
        if self.getVerbose():
            general_utils.printSeparation("*", 50)
            print("[INFO] - Start runTraining function.")
            print("[INFO] - Getting model {0} with {1} optimizer...".format(self.name, self.optimizerInfo["name"]))

        # based on the number of GPUs availables
        # call the function called self.name in models.py
        if n_gpu==1:
            self.model = getattr(models, self.name)(params=self.params, to_categ=self.to_categ)
        else:
            # TODO: problems during the load of the model (?)
            with tf.device('/cpu:0'):
                self.model = getattr(models, self.name)(params=self.params, to_categ=self.to_categ)
            self.model = multi_gpu_model(self.model, gpus=n_gpu)

        if self.getVerbose() and self.summaryFlag==0:
            print(self.model.summary())

            plot_model(
                self.model,
                to_file=general_utils.getFullDirectoryPath(self.savedModelFolder)+self.getNNID("model")+".png",
                show_shapes=True,
                rankdir='LR'
            )
            self.summaryFlag+=1

        # check if the model has some saved weights to load...
        self.initial_epoch = 0
        if self.arePartialWeightsSaved(p_id):
            self.loadModelFromPartialWeights(p_id)

        # compile the model with optimizer, loss function and metrics
        self.compileModel()

        self.sample_weights = self.getSampleWeights("train")
        # Set the callbacks
        self.setCallbacks(p_id, self.sample_weights)


################################################################################
# Run the training over the dataset based on the model
    def runTraining(self, p_id, n_gpu):
        self.initializeTraining(p_id, n_gpu)

        self.dataset["train"]["labels"] = dataset_utils.getLabelsFromIndex(train_df=self.train_df, dataset=self.dataset["train"], modelname=self.name, to_categ=self.to_categ, flag="train")
        self.dataset["val"]["labels"] = None if self.val["validation_perc"]==0 else dataset_utils.getLabelsFromIndex(train_df=self.train_df, dataset=self.dataset["val"], modelname=self.name, to_categ=self.to_categ, flag="val")
        if self.supervised: self.dataset["test"]["labels"] = dataset_utils.getLabelsFromIndex(train_df=self.train_df, dataset=self.dataset["test"], modelname=self.name, to_categ=self.to_categ, flag="test")

        # fit and train the model
        self.train = training.fitModel(
            model=self.model,
            dataset=self.dataset,
            batch_size=self.batch_size,
            epochs=self.epochs,
            listOfCallbacks=self.callbacks,
            sample_weights=self.sample_weights,
            initial_epoch=self.initial_epoch,
            save_activation_filter=self.save_activation_filter,
            intermediate_activation_path=self.intermediateActivationFolder,
            use_multiprocessing=self.mp)

        # plot the loss and accuracy of the training
        training.plotLossAndAccuracy(self, p_id)

        # deallocate memory
        for flag in ["train", "val", "test"]:
            for type in ["labels", "data"]:
                if type in self.dataset[flag]: del self.dataset[flag][type]

################################################################################
# Run the training for every batch and LOAD only the necessary dataset (one patient at the time)
    def runTrainingOnBatch(self, p_id, n_gpu):
        suffix = general_utils.getSuffix() # es == "_4_16x16"
        dict_metrics = list()
        filename = self.getSavedInformation(p_id, path=self.savePartialModelFolder)

        if self.getVerbose():
            general_utils.printSeparation("*", 50)
            print("[WARNING] - Using train_on_batch flag!")
            print("[INFO] - Start runTrainingOnBatch function.")

        if self.getVerbose():
            general_utils.printSeparation("-", 50)
            print("[INFO] - Getting model {0} with {1} optimizer...".format(self.name, self.optimizerInfo["name"]))

        # based on the number of GPUs availables
        # call the function called self.name in models.py
        if n_gpu==1:
            self.model = getattr(models, self.name)(self.dataset["val"]["data"], params=self.params, to_categ=self.to_categ)
        else:
            # TODO: problems during the load of the model (?)
            with tf.device('/cpu:0'):
                self.model = getattr(models, self.name)(self.dataset["val"]["data"], params=self.params, to_categ=self.to_categ)
            self.model = multi_gpu_model(self.model, gpus=n_gpu)

        if self.getVerbose():
            print(self.model.summary())

        # plot_model(
        #     self.model,
        #     to_file=general_utils.getFullDirectoryPath(self.savedModelFolder)+self.getNNID("model")+".png",
        #     show_shapes=True,
        #     rankdir='LR'
        # )

        # check if the model has some saved weights to load...
        self.initial_epoch = 0
        if self.arePartialWeightsSaved(p_id): self.loadModelFromPartialWeights(p_id)

        # compile the model with optimizer, loss function and metrics
        self.compileModel()

        sample_weights = self.getSampleWeights("train")

        # Set the callbacks
        self.setCallbacks(p_id, sample_weights)

        self.dataset["val"]["labels"] = None if self.val["validation_perc"]==0 else dataset_utils.getLabelsFromIndex(train_df=self.train_df, dataset=self.dataset["val"], modelname=self.name, to_categ=self.to_categ, flag="val")
        if self.supervised: self.dataset["test"]["labels"] = dataset_utils.getLabelsFromIndex(train_df=self.train_df, dataset=self.dataset["test"], modelname=self.name, to_categ=self.to_categ, flag="test")

        for metric_name in self.model.metrics_names: dict_metrics.append({"name":metric_name, "val":list()})

        for epoch in range(self.initial_epoch, self.epochs):
            print("Epoch: {}".format(epoch+1))
            count_trained_elements = 0

            while count_trained_elements<len(self.dataset["train"]["indices"]): # going over all the element in the dataset
                indices = self.dataset["train"]["indices"][count_trained_elements:count_trained_elements+self.batch_size]
                x = [a for a in np.array(self.train_df.pixels.values[indices], dtype=object)]
                y = [a for a in np.array(self.train_df.ground_truth.values[indices])]
                if type(x) is not np.array: x = np.array(x)
                if type(y) is not np.array: y = np.array(y)
                y = y.astype("float32")
                y /= 255 # convert the label in [0, 1] values

                w = sample_weights[count_trained_elements:count_trained_elements+self.batch_size]

                self.model, ret = training.trainOnBatch(self.model, x, y, w)

                for i, r in enumerate(ret): dict_metrics[i]["val"].append(r)
                if count_trained_elements%200==0 and count_trained_elements!=0: print(ret) #print the values every 200 batches

                count_trained_elements += self.batch_size

            if (epoch+1)%2==0 and epoch!=0: self.model.save_weights(filename+constants.suffix_partial_weights+"{:02d}.h5".format(epoch)) # save the model weights every 2 epochs

            tmpSavedModels = glob.glob(filename+constants.suffix_partial_weights+"*.h5")
            if len(tmpSavedModels) > 1: # just to be sure and not delete everything
                for file in tmpSavedModels:
                    if filename+constants.suffix_partial_weights in file:
                        tmpEpoch = general_utils.getEpochFromPartialWeightFilename(file)
                        if tmpEpoch < epoch: # Remove the old saved weights
                            os.remove(file)

        # plot the loss and accuracy of the training
        training.plotMetrics(self, p_id, dict_metrics)
        # save the activation filters
        if self.save_activation_filter: training.saveActivationFilter(self.model, shape=tuple(self.train_df.pixels.values[0].shape), intermediate_activation_path=self.intermediateActivationFolder)

################################################################################
#
    def prepareSequenceClass(self):
        # train data sequence
        self.train_sequence = sequence_utils.trainValSequence(self.train_df,
            self.dataset["train"]["indices"], self.getSampleWeights("train"),
            "pixels", "ground_truth", self.batch_size)
        # validation data sequence
        self.val_sequence = sequence_utils.trainValSequence(self.train_df,
            self.dataset["val"]["indices"], self.getSampleWeights("val"),
            "pixels", "ground_truth", self.batch_size)

################################################################################
#
    def runTrainSequence(self, p_id, n_gpu):
        self.initializeTraining(p_id, n_gpu)

        self.train = training.fit_generator(
            model=self.model,
            train_sequence=self.train_sequence,
            val_sequence=self.val_sequence,
            epochs=self.epochs,
            listOfCallbacks=self.callbacks,
            initial_epoch=self.initial_epoch,
            save_activation_filter=self.save_activation_filter,
            use_multiprocessing=self.mp)

        # plot the loss and accuracy of the training
        training.plotLossAndAccuracy(self, p_id)

################################################################################
# Get the sample weight from the dataset
    def getSampleWeights(self, flagDataset):
        self.N_BACKGROUND, self.N_BRAIN, self.N_PENUMBRA, self.N_CORE, self.N_TOT = dataset_utils.getNumberOfElements(self.train_df)

        if constants.N_CLASSES==4:
            if constants.getM()>=512: # we have only one label
                # function that map each PIXELVALUES[2] with 150, PIXELVALUES[3] with 20 and the rest with 0.1 and sum them
                f = lambda x : np.sum(np.where(np.array(x)==constants.PIXELVALUES[2],150,np.where(np.array(x)==constants.PIXELVALUES[3],20,0.1)))

                sample_weights = self.train_df.ground_truth.map(f)
                sample_weights = sample_weights/(constants.getM()*constants.getN())
            else:
                # see: "ISBI 2019 C-NMC Challenge: Classification in Cancer Cell Imaging" section 4.1 pag 68
                sample_weights = self.train_df.label.map({
                    constants.LABELS[0]: self.N_TOT/(constants.N_CLASSES*self.N_BACKGROUND) if self.N_BACKGROUND>0 else 0, # N_TOT/N_BACKGROUND,
                    constants.LABELS[1]: self.N_TOT/(constants.N_CLASSES*self.N_BRAIN) if self.N_BRAIN>0 else 0, # N_TOT/N_BRAIN,
                    constants.LABELS[2]: self.N_TOT/(constants.N_CLASSES*self.N_PENUMBRA) if self.N_PENUMBRA>0 else 0, # N_TOT/N_PENUMBRA,
                    constants.LABELS[3]: self.N_TOT/(constants.N_CLASSES*self.N_CORE) if self.N_CORE>0 else 0, # N_TOT/N_CORE
                })
        elif constants.N_CLASSES==3:
            if constants.getM()>=512: # we have only one label
                # function that map each PIXELVALUES[2] with 150, PIXELVALUES[3] with 20 and the rest with 0.1 and sum them
                f = lambda x : np.sum(np.where(np.array(x)==150,150,np.where(np.array(x)==76,20,0.1)))

                sample_weights = self.train_df.ground_truth.map(f)
                sample_weights = sample_weights/(constants.getM()*constants.getN())
            else:
                # see: "ISBI 2019 C-NMC Challenge: Classification in Cancer Cell Imaging" section 4.1 pag 68
                sample_weights = self.train_df.label.map({
                    constants.LABELS[0]: self.N_TOT/(constants.N_CLASSES*(self.N_BACKGROUND+self.N_BRAIN)) if self.N_BACKGROUND+self.N_BRAIN>0 else 0, # N_TOT/N_BACKGROUND,
                    constants.LABELS[1]: self.N_TOT/(constants.N_CLASSES*self.N_PENUMBRA) if self.N_PENUMBRA>0 else 0, # N_TOT/N_PENUMBRA,
                    constants.LABELS[2]: self.N_TOT/(constants.N_CLASSES*self.N_CORE) if self.N_CORE>0 else 0, # N_TOT/N_CORE
                })
        else: # we are in a binary class problem
            f = lambda x : np.sum(np.array(x))
            sample_weights = self.train_df.ground_truth.map(f)

        return np.array(sample_weights.values[self.dataset[flagDataset]["indices"]])

################################################################################
# Save the trained model and its relative weights
    def saveModelAndWeight(self, p_id):
        saved_modelname = self.getSavedModel(p_id)
        saved_weightname = self.getSavedWeight(p_id)

        p_id = general_utils.getStringFromIndex(p_id)
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(saved_modelname, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(saved_weightname)

        if self.getVerbose():
            general_utils.printSeparation("-", 50)
            print("[INFO - Saving] - Saved model and weights to disk!")

################################################################################
# Call the function located in testing for predicting and saved the images
    def predictAndSaveImages(self, p_id):
        if self.getVerbose():
            general_utils.printSeparation("+", 50)
            print("[INFO] - Executing function: predictAndSaveImages for patient {}".format(p_id))

        stats = testing.predictAndSaveImages(self, p_id)
        if self.save_statistics: self.saveStats(stats, p_id)

        return stats

################################################################################
# Function to save in a file the statistic for the test patients
    def saveStats(self, stats, p_id):
        suffix = general_utils.getSuffix()
        if p_id == "ALL":
            with open(general_utils.getFullDirectoryPath(self.saveTextFolder)+self.getNNID(p_id)+suffix+".txt", "a+") as text_file:
                text_file.write("====================================================\n")
                text_file.write("====================================================\n")
                for func in self.statistics:
                    for classToEval in self.classes_to_evaluate:
                        if self.epsiloList[0]!=None:
                            for idxE, epsilons in enumerate(self.epsiloList):
                                tn = sum(cm[0] for cm in stats[func.__name__][classToEval][idxE])
                                fn = sum(cm[1] for cm in stats[func.__name__][classToEval][idxE])
                                fp = sum(cm[2] for cm in stats[func.__name__][classToEval][idxE])
                                tp = sum(cm[3] for cm in stats[func.__name__][classToEval][idxE])
                                res = func(tn,fn,fp,tp)
                                standard_dev = 0
                                # meanV = np.mean(stats[func.__name__][classToEval])
                                # stdV = np.std(stats[func.__name__][classToEval])
                                #text_file.write("\n\n EPSILONS: *{0} **{1} ***{2} ... {3} idx  \n".format(epsilons[0], epsilons[1], epsilons[2], idxE))
                                text_file.write("TEST MEAN {0} {1}: {2} \n".format(func.__name__, classToEval, round(float(res), 3)))
                                text_file.write("TEST STD {0} {1}: {2} \n".format(func.__name__, classToEval, round(float(standard_dev), 3)))
                            # text_file.write("TEST MEAN %s %s: %.2f%% \n" % (func.__name__, classToEval, round(meanV,6)*100))
                            # text_file.write("TEST STD %s %s: %.2f \n" % (func.__name__, classToEval, round(stdV,6)))
                    text_file.write("----------------------------------------------------- \n")
        else:
            for func in self.statistics:
                for classToEval in self.classes_to_evaluate:
                    if self.epsiloList[0]!=None:
                        for idxE, epsilons in enumerate(self.epsiloList):
                            tn = sum(cm[0] for cm in stats[func.__name__][classToEval][idxE])
                            fn = sum(cm[1] for cm in stats[func.__name__][classToEval][idxE])
                            fp = sum(cm[2] for cm in stats[func.__name__][classToEval][idxE])
                            tp = sum(cm[3] for cm in stats[func.__name__][classToEval][idxE])

                            res = func(tn,fn,fp,tp)
                            #print("EPSILONS: *{0} **{1} ***{2} ... {3}\%  \n".format(epsilons[0], epsilons[1], epsilons[2], idxE))
                            print("TEST {0} {1}: {2}".format(func.__name__, classToEval, round(float(res), 3)))

################################################################################
# Test the model with the selected patient (if the number of patient to test is > 0)
    def evaluateModelWithCategorics(self, p_id, isAlreadySaved):
        if self.getVerbose():
            general_utils.printSeparation("+", 50)
            print("[INFO] - Evaluating the model for patient {}".format(p_id))

        self.testing_score = testing.evaluateModel(self, p_id, isAlreadySaved)

################################################################################
# set the flag for single/multi PROCESSING
    def setProcessingEnv(self, mp):
        self.mp = mp

################################################################################
# return the saved model or weight (based on the suffix)
    def getSavedInformation(self, p_id, path, other_info="", suffix=""):
        # mJ-Net_DA_ADAM_4_16x16.json <-- example weights name
        # mJ-Net_DA_ADAM_4_16x16.h5 <-- example model name
        path = general_utils.getFullDirectoryPath(path)+self.getNNID(p_id)+other_info+general_utils.getSuffix()
        return path+suffix

################################################################################
# return the saved model
    def getSavedModel(self, p_id):
        return self.getSavedInformation(p_id, path=self.savedModelFolder, suffix=".json")

################################################################################
# return the saved weight
    def getSavedWeight(self, p_id):
        return self.getSavedInformation(p_id, path=self.savedModelFolder, suffix=".h5")

################################################################################
# return NeuralNetwork ID
    def getNNID(self, p_id):
        # CAREFUL WITH THIS
        if self.OVERRIDE_MODELS_ID_PATH:
            # needs to override the model id to use a different model to test various patients
            id = self.OVERRIDE_MODELS_ID_PATH
        else:
            id = self.name
            if self.da: id += "_DA"
            id += ("_"+self.optimizerInfo["name"].upper())

            id += ("_VAL"+str(self.val["validation_perc"]))
            if self.val["random_validation_selection"]: id += ("_RANDOM")

            if self.to_categ: id += ("_SOFTMAX") # differenciate between softmax and sigmoid last activation layer

            # if there is cross validation, add the PATIENT_ID to differenciate the models
            if self.cross_validation:
                id += ("_" + p_id)

        return id

################################################################################
# return the verbose flag
    def getVerbose(self):
        return constants.getVerbose()
