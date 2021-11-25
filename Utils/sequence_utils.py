import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import math, cv2, glob, time, os
from typing import Set, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import ndimage
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from Model import constants
from Utils import general_utils, dataset_utils, model_utils


################################################################################
# https://faroit.com/keras-docs/2.1.3/models/sequential/#fit_generator
class datasetSequence(Sequence):
    def __init__(self, dataframe, indices, sample_weights, x_label, y_label, multiInput, batch_size, params, back_perc, is3dot5DModel,
                 is4DModel, inputImgFlag, supervised, patientsFolder, SVO_focus=False, flagtype="train", loss=None):
        self.indices = indices
        self.dataframe = dataframe.iloc[self.indices]

        self.sample_weights = sample_weights
        self.x_label = x_label
        self.y_label = y_label
        self.multiInput = multiInput
        self.batch_size = batch_size
        self.params = params
        self.back_perc = back_perc
        self.flagtype = flagtype
        self.loss = loss
        self.is3dot5DModel = is3dot5DModel
        self.is4DModel = is4DModel
        self.SVO_focus = SVO_focus
        self.inputImgFlag = inputImgFlag  # only works when the input are the PMs (concatenate)
        self.supervised = supervised
        self.patientsFolder = patientsFolder

        if self.flagtype != "test":
            # get ALL the rows with label != from background
            self.dataframenoback = self.dataframe.loc[self.dataframe.label != constants.LABELS[0]]
            # also, get a back_perc of rows with label == background
            self.dataframeback = self.dataframe.loc[self.dataframe.label == constants.LABELS[0]]
            if self.back_perc < 100: self.dataframeback = self.dataframeback[:int((len(self.dataframeback)/100)*self.back_perc)]
            # combine the two dataframes
            self.dataframe = pd.concat([self.dataframenoback, self.dataframeback], sort=False)

        self.index_pd_DA: Dict[str, Set[Any]] = {"0": set(),"1": set(),"2": set(),"3": set(),"4": set(),"5": set()}
        self.index_batch = None

        self.n_slices = 0 if "n_slices" not in self.params.keys() else self.params["n_slices"]

    def on_epoch_end(self):
        self.dataframe = self.dataframe.sample(frac=1)  # shuffle the dataframe rows at the end of a epoch

    # Every Sequence must implement the __getitem__ and the __len__ methods
    def __len__(self):
        return math.ceil(len(self.dataframe) / self.batch_size)

    def __getitem__(self, idx):
        start = idx*self.batch_size
        end = (idx+1)*self.batch_size

        current_batch = self.dataframe[start:end]
        self.index_batch = current_batch.index

        # empty initialization
        X = np.empty((len(current_batch), constants.getM(), constants.getN(), constants.NUMBER_OF_IMAGE_PER_SECTION, 1)) if constants.getTIMELAST() else np.empty((len(current_batch), constants.NUMBER_OF_IMAGE_PER_SECTION, constants.getM(), constants.getN(), 1))
        Y = np.empty((len(current_batch), constants.getM(), constants.getN()))
        weights = np.empty((len(current_batch),))

        if constants.getUSE_PM(): X = np.empty((len(current_batch), constants.getM(), constants.getN()))

        # reset the index for the data augmentation
        self.index_pd_DA: Dict[str, Set[Any]] = {"0": set(),"1": set(),"2": set(),"3": set(),"4": set(),"5": set()}

        # path to the folder containing the NUMBER_OF_IMAGE_PER_SECTION time point images
        X, weights = self.getX(X, current_batch, weights)
        # path to the ground truth image
        if self.y_label=="ground_truth": Y, weights = self.getY(current_batch, weights)

        return X, Y, weights

    ################################################################################
    # return the X set and the relative weights based on the pixels column
    def getX(self, X, current_batch, weights):
        for index, (_, row) in enumerate(current_batch.iterrows()):
            current_folder = row[self.x_label]
            # add the index into the correct set
            self.index_pd_DA[str(row["data_aug_idx"])].add(index)
            X = model_utils.getCorrectXForInputModel(self, current_folder, row, batchIndex=index, batch_length=len(current_batch), X=X, train=True)

        return X, weights

    ################################################################################
    # Return the Y set and the weights
    def getY(self, current_batch, weights):
        Y = []
        for aug_idx in self.index_pd_DA.keys():
            if len(self.index_pd_DA[aug_idx])==0: continue
            for index in self.index_pd_DA[aug_idx]:
                row_index = self.index_batch[index]
                filename = current_batch.loc[row_index][self.y_label]
                coord = current_batch.loc[row_index]["x_y"]  # coordinates of the slice window
                if not isinstance(filename,str): print(filename)
                img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
                assert img is not None, "The image {} is None".format(filename)
                img = general_utils.getSlicingWindow(img, coord[0], coord[1], isgt=True)

                # remove the brain from the image ==> it becomes background
                if constants.N_CLASSES<=3: img[img == 85] = constants.PIXELVALUES[0]
                # remove the penumbra ==> it becomes core
                if constants.N_CLASSES==2: img[img == 170] = constants.PIXELVALUES[1]

                # Override the weights based on the pixel values
                if constants.N_CLASSES>2:
                    core_idx, penumbra_idx = 3, 2
                    if constants.N_CLASSES == 3: core_idx, penumbra_idx = 2, 1
                    core_value, core_weight = constants.PIXELVALUES[core_idx], constants.HOT_ONE_WEIGHTS[0][core_idx]
                    penumbra_value, penumbra_weight = constants.PIXELVALUES[penumbra_idx], constants.HOT_ONE_WEIGHTS[0][penumbra_idx]

                    # focus on the SVO core only during training (only for SUS2020 dataset)!
                    if self.SVO_focus and current_batch.loc[row_index]["severity"]=="02":
                        core_weight *= 6
                        penumbra_weight *= 6

                    # sum the pixel value for the image with the corresponding "weight" for class
                    f = lambda x: np.sum(np.where(np.array(x) == core_value, core_weight,
                                                  np.where(np.array(x) == penumbra_value, penumbra_weight, constants.HOT_ONE_WEIGHTS[0][0])))
                    weights[index] = f(img) / (constants.getM() * constants.getN())
                elif constants.N_CLASSES == 2:
                    core_value, core_weight = constants.PIXELVALUES[1], constants.HOT_ONE_WEIGHTS[0][1]
                    f = lambda x: np.sum(np.where(np.array(x) == core_value, core_weight, constants.HOT_ONE_WEIGHTS[0][0]))
                    weights[index] = f(img) / (constants.getM() * constants.getN())

                # convert the label in [0, 1] values,
                # for to_categ the division happens inside dataset_utils.getSingleLabelFromIndexCateg
                if not constants.getTO_CATEG(): img = np.divide(img,255)

                if aug_idx=="0": img = img if not constants.getTO_CATEG() or self.loss=="sparse_categorical_crossentropy" else dataset_utils.getSingleLabelFromIndexCateg(img)
                elif aug_idx=="1": img = np.rot90(img) if not constants.getTO_CATEG() or self.loss=="sparse_categorical_crossentropy" else dataset_utils.getSingleLabelFromIndexCateg(np.rot90(img))
                elif aug_idx=="2": img = np.rot90(img, 2) if not constants.getTO_CATEG() or self.loss=="sparse_categorical_crossentropy" else dataset_utils.getSingleLabelFromIndexCateg(np.rot90(img, 2))
                elif aug_idx=="3": img = np.rot90(img, 3) if not constants.getTO_CATEG() or self.loss=="sparse_categorical_crossentropy" else dataset_utils.getSingleLabelFromIndexCateg(np.rot90(img, 3))
                elif aug_idx=="4": img = np.flipud(img) if not constants.getTO_CATEG() or self.loss=="sparse_categorical_crossentropy" else dataset_utils.getSingleLabelFromIndexCateg(np.flipud(img))
                elif aug_idx=="5": img = np.fliplr(img) if not constants.getTO_CATEG() or self.loss=="sparse_categorical_crossentropy" else dataset_utils.getSingleLabelFromIndexCateg(np.fliplr(img))

                Y.append(img)

        return np.array(Y), weights
