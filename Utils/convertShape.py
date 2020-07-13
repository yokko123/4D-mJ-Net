import glob, time
import numpy as np
import pandas as pd
import pickle as pkl

DATASET_NAME = "SUS2020_v2/"
SCRIPT_PATH = "/local/home/lucat/DATASET/"+DATASET_NAME

IMAGE_WIDTH, IMAGE_HEIGHT = 512, 512
TILE_DIVISION = 32
VERBOSE = 1
M, N = int(IMAGE_WIDTH/TILE_DIVISION), int(IMAGE_HEIGHT/TILE_DIVISION)
SLICING_PIXELS = int(M/4) # USE ALWAYS M/4

################################################################################
# Return the elements in the filename saved as a pickle
def readFromPickle(filename):
    file = open(filename, "rb")
    return pkl.load(file)

################################################################################

def convertDatasets():
    suffix_filename = "_"+str(SLICING_PIXELS)+"_"+str(M)+"x"+str(N)
    datasetFolder = glob.glob(SCRIPT_PATH+"*"+suffix_filename+"*")

    for index, df_filename in enumerate(datasetFolder): # for each dataframe patient
        print("[INFO] - Analyzing {0}/{1}; patient dataframe: {2}...".format(index+1, len(datasetFolder), df_filename))
        df = readFromPickle(df_filename)

        start = time.time()

        if df.pixels[0].shape == (M,N,30): continue

        for row in df.itertuples():
            pixels = row.pixels
            new_pixels = np.empty((M,N,1))

            for time_pixel in pixels:
                time_pixel = time_pixel.reshape(time_pixel.shape[0],time_pixel.shape[1],1)
                new_pixels = np.append(new_pixels, time_pixel, axis=2)

            new_pixels = np.delete(new_pixels,0,axis=2)  # remove the first element (generate by np.empty)
            df.pixels[row.Index] = new_pixels

        f = open(df_filename, 'wb')
        pkl.dump(df, f)

        end = time.time()
        print("Time: {0}s".format(round(end-start, 3)))

################################################################################
# ## Main
################################################################################
if __name__ == '__main__':
    start = time.time()
    print("Converting dataset...")
    convertDatasets()
    end = time.time()
    print("Total time: {0}s".format(round(end-start, 3)))
