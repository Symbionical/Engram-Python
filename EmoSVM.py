from operator import concat
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided as ast
from pandas.core.construction import array
from pandas.core.frame import DataFrame
from pandas.io.pytables import incompatibility_doc
from sklearn import svm
from scipy.io import loadmat
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams
from brainflow.exit_codes import *
from sklearn.utils import class_weight

########################################################################
# ######################################################################
# Initialise some settings for pre-processing the data 
print("Initialising pre-processing settings")

board_id = BoardIds.CYTON_BOARD.value

# BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
# MLModel.enable_ml_logger()

board = BoardShim(board_id, params)
sampling_rate = BoardShim.get_sampling_rate(board_id)
eeg_channels = [0, 1, 2, 3, 4, 5, 6, 7]
nfft = DataFilter.get_nearest_power_of_two(sampling_rate)

########################################################################
# FUNCTIONS ############################################################

# Data manipulation functions
print("Initializing functions")

# This function condenses labels into two classes 
# The label coding is as follows 
# 1,6-Sad
# 2-Neutral
# 3,5-Happy
# 4-Baseline

# recodes data labels
def code_labels(coded_df):
    
    # Positive Emotions
    coded_df = coded_df.replace(to_replace = 3, value= 8)    
    coded_df = coded_df.replace(to_replace = 5, value= 8)

    
    # Negative Emotions
    coded_df = coded_df.replace(to_replace = 1, value= 9)    
    coded_df = coded_df.replace(to_replace = 6, value= 9)

    return coded_df

# turns data labels to descriptive strings
def stringify_classes(df_to_stringify):

    df_to_stringify = df_to_stringify.replace(to_replace = 8, value = "Positive")
    df_to_stringify = df_to_stringify.replace(to_replace = 9, value = "Negative")

    return df_to_stringify

# Groups dataframe into smaller dataframes of n chunks
def chunk_df(_frame_to_chunk, n):
    chunked_frame = [_frame_to_chunk[i:i+n] for i in range(0,Mainframe.shape[0],n)]
    _clean_chunked_frame = []
    for frame in chunked_frame:
        if frame[8].iloc[0] == frame[8].iloc[-1] and len(frame) == n:
            _clean_chunked_frame.append(frame)

    return _clean_chunked_frame

# Filters EEG data
def filter_signal(_data, _eeg_channels):
    for channel in _eeg_channels:
        #0hz - 75hz bandpass
        DataFilter.perform_bandpass(_data[channel], BoardShim.get_sampling_rate(board_id), 37.5, 75, 4, FilterTypes.BESSEL.value, 0)
        # 50hz filter
        DataFilter.perform_bandstop(_data[channel], BoardShim.get_sampling_rate(board_id), 50, 1, 3, FilterTypes.BUTTERWORTH.value, 0)
        #Denoise
        DataFilter.perform_wavelet_denoising(_data[channel], 'coif3', 3)
    return _data

# returns bandpowers averaged across all channels
def get_avg_band_df(_clean_chunked_frame, eeg_channels, sampling_rate):
    
    bands_to_array = []

    for frame in _clean_chunked_frame:
        chunked_eeg_data = frame[[0,1,2,3,4,5,6,7]].to_numpy()
        chunked_eeg_data = np.transpose(chunked_eeg_data)
        filtered_eeg_data = filter_signal(chunked_eeg_data, eeg_channels)
        bands = DataFilter.get_avg_band_powers(filtered_eeg_data, eeg_channels, sampling_rate, True)
        bands_to_array.append(bands[0])

    bands_to_array = np.concatenate([bands_to_array], axis=0)
    _avg_bands_df = pd.DataFrame(bands_to_array)

    band_labels = []

    for frame in _clean_chunked_frame:
        band_labels.append(frame[8].iloc[0])

    band_labels = np.concatenate([band_labels], axis=0)
    _avg_bands_df[5] = band_labels

    return _avg_bands_df

# returns bandpowers for each channel
def get_bands_df(_clean_chunked_frame, eeg_channels, sampling_rate):
    
    bands_to_array = []

    for frame in _clean_chunked_frame: 
        chunked_eeg_data = frame[[0,1,2,3,4,5,6,7]].to_numpy()
        chunked_eeg_data = np.transpose(chunked_eeg_data)
        filtered_eeg_data = filter_signal(chunked_eeg_data, eeg_channels)

        row_of_bands = []

        fp1_bands = DataFilter.get_avg_band_powers(filtered_eeg_data, [0], sampling_rate, True)
        fp2_bands = DataFilter.get_avg_band_powers(filtered_eeg_data, [1], sampling_rate, True)
        f3_bands = DataFilter.get_avg_band_powers(filtered_eeg_data, [2], sampling_rate, True)
        f4_bands = DataFilter.get_avg_band_powers(filtered_eeg_data, [3], sampling_rate, True)
        f7_bands = DataFilter.get_avg_band_powers(filtered_eeg_data, [4], sampling_rate, True)
        f8_bands = DataFilter.get_avg_band_powers(filtered_eeg_data, [5], sampling_rate, True)
        t7_bands = DataFilter.get_avg_band_powers(filtered_eeg_data, [6], sampling_rate, True)
        t8_bands = DataFilter.get_avg_band_powers(filtered_eeg_data, [7], sampling_rate, True)

        row_of_bands = [fp1_bands[0], fp2_bands[0], f3_bands[0], f4_bands[0], f7_bands[0], f8_bands[0], t7_bands[0], t8_bands[0]]
        row_of_bands = np.concatenate([row_of_bands], axis=None)
        bands_to_array.append(row_of_bands)

    bands_to_array = np.concatenate([bands_to_array], axis=0)
    _bands_df = pd.DataFrame(bands_to_array)

    band_labels = []

    for frame in _clean_chunked_frame:
        band_labels.append(frame[8].iloc[0])

    band_labels = np.concatenate([band_labels], axis=0)
    _bands_df[40] = band_labels

    return _bands_df

###########################################################################
###########################################################################

# DATA PROCESSING SCRIPT ##################################################
###########################################################################

# Load the Files
print("Loading data ...")

mat1 = loadmat("DataSets/LUMED/EEG_GSR_Data/s01.mat")
mat2 = loadmat("DataSets/LUMED/EEG_GSR_Data/s02.mat")
mat3 = loadmat("DataSets/LUMED/EEG_GSR_Data/s03.mat")
mat4 = loadmat("DataSets/LUMED/EEG_GSR_Data/s04.mat")
mat5 = loadmat("DataSets/LUMED/EEG_GSR_Data/s05.mat")
mat6 = loadmat("DataSets/LUMED/EEG_GSR_Data/s06.mat")
mat7 = loadmat("DataSets/LUMED/EEG_GSR_Data/s07.mat")
mat8 = loadmat("DataSets/LUMED/EEG_GSR_Data/s08.mat")
mat9 = loadmat("DataSets/LUMED/EEG_GSR_Data/s09.mat")
mat10 = loadmat("DataSets/LUMED/EEG_GSR_Data/s10.mat")
mat11 = loadmat("DataSets/LUMED/EEG_GSR_Data/s11.mat")
mat12 = loadmat("DataSets/LUMED/EEG_GSR_Data/s12.mat")
mat13 = loadmat("DataSets/LUMED/EEG_GSR_Data/s13.mat")

# Extact the data, ingoring matlab meta data
print("Extracting Values")

struct1 = mat1["EEGData"]
struct2 = mat2["EEGData"]
struct3 = mat3["EEGData"]
struct4 = mat4["EEGData"]
struct5 = mat5["EEGData"]
struct6 = mat6["EEGData"]
struct7 = mat7["EEGData"]
struct8 = mat8["EEGData"]
struct9 = mat9["EEGData"]
struct10 = mat10["EEGData"]
struct11 = mat11["EEGData"]
struct12 = mat12["EEGData"]
struct13 = mat13["EEGData"]

# Deconstruct the matlab struct into its constituents 
print("Deconstructing structs")

data1 = struct1[0,0]
data2 = struct2[0,0]
data3 = struct3[0,0]
data4 = struct4[0,0]
data5 = struct5[0,0]
data6 = struct6[0,0]
data7 = struct7[0,0]
data8 = struct8[0,0]
data9 = struct9[0,0]
data10 = struct10[0,0]
data11 = struct11[0,0]
data12 = struct12[0,0]
data13 = struct13[0,0]

# Convert them all into dataframes
print("Constructing EEG dataframes")

EEG_data_df1 = pd.DataFrame(data1['Data'])
EEG_data_df2 = pd.DataFrame(data2['Data'])
EEG_data_df3 = pd.DataFrame(data3['Data'])
EEG_data_df4 = pd.DataFrame(data4['Data'])
EEG_data_df5 = pd.DataFrame(data5['Data'])
EEG_data_df6 = pd.DataFrame(data6['Data'])
EEG_data_df7 = pd.DataFrame(data7['Data'])
EEG_data_df8 = pd.DataFrame(data8['Data'])
EEG_data_df9 = pd.DataFrame(data9['Data'])
EEG_data_df10 = pd.DataFrame(data10['Data'])
EEG_data_df11 = pd.DataFrame(data11['Data'])
EEG_data_df12 = pd.DataFrame(data12['Data'])
EEG_data_df13 = pd.DataFrame(data13['Data'])

# Convert them all into dataframes
print("Constructing label dataframes")

Label_data_df1 = pd.DataFrame(data1['Labels'].astype(np.int32))
Label_data_df2 = pd.DataFrame(data2['Labels'].astype(np.int32))
Label_data_df3 = pd.DataFrame(data3['Labels'].astype(np.int32))
Label_data_df4 = pd.DataFrame(data4['Labels'].astype(np.int32))
Label_data_df5 = pd.DataFrame(data5['Labels'].astype(np.int32))
Label_data_df6 = pd.DataFrame(data6['Labels'].astype(np.int32))
Label_data_df7 = pd.DataFrame(data7['Labels'].astype(np.int32))
Label_data_df8 = pd.DataFrame(data8['Labels'].astype(np.int32))
Label_data_df9 = pd.DataFrame(data9['Labels'].astype(np.int32))
Label_data_df10 = pd.DataFrame(data10['Labels'].astype(np.int32))
Label_data_df11 = pd.DataFrame(data11['Labels'].astype(np.int32))
Label_data_df12 = pd.DataFrame(data12['Labels'].astype(np.int32))
Label_data_df13 = pd.DataFrame(data13['Labels'].astype(np.int32))

# Downsample EEG data to 250hz
print("Downsampling EEG data")

EEG_data_df1 = EEG_data_df1.iloc[::2,:]
EEG_data_df2 = EEG_data_df2.iloc[::2,:]
EEG_data_df3 = EEG_data_df3.iloc[::2,:]
EEG_data_df4 = EEG_data_df4.iloc[::2,:]
EEG_data_df5 = EEG_data_df5.iloc[::2,:]
EEG_data_df6 = EEG_data_df6.iloc[::2,:]
EEG_data_df7 = EEG_data_df7.iloc[::2,:]
EEG_data_df8 = EEG_data_df8.iloc[::2,:]
EEG_data_df9 = EEG_data_df9.iloc[::2,:]
EEG_data_df10 = EEG_data_df10.iloc[::2,:]
EEG_data_df11 = EEG_data_df11.iloc[::2,:]
EEG_data_df12 = EEG_data_df12.iloc[::2,:]
EEG_data_df13 = EEG_data_df13.iloc[::2,:]

# Downsample Labels to 250hz
print ("Downsampling labels")

Label_data_df1 = Label_data_df1.iloc[::2,:]
Label_data_df2 = Label_data_df2.iloc[::2,:]
Label_data_df3 = Label_data_df3.iloc[::2,:]
Label_data_df4 = Label_data_df4.iloc[::2,:]
Label_data_df5 = Label_data_df5.iloc[::2,:]
Label_data_df6 = Label_data_df6.iloc[::2,:]
Label_data_df7 = Label_data_df7.iloc[::2,:]
Label_data_df8 = Label_data_df8.iloc[::2,:]
Label_data_df9 = Label_data_df9.iloc[::2,:]
Label_data_df10 = Label_data_df10.iloc[::2,:]
Label_data_df11 = Label_data_df11.iloc[::2,:]
Label_data_df12 = Label_data_df12.iloc[::2,:]
Label_data_df13 = Label_data_df13.iloc[::2,:]

# Group the EEG dataframes together and downsample to 250hz
print("Grouping EEG dataframes")

EEG_frames = [
    EEG_data_df1,
    EEG_data_df2,
    EEG_data_df3,
    EEG_data_df4,
    EEG_data_df5,
    EEG_data_df6,
    EEG_data_df7,
    EEG_data_df8,
    EEG_data_df9,
    EEG_data_df10,
    EEG_data_df11,
    EEG_data_df12,
    EEG_data_df13
]

# Group the label dataframes together and downsample to 250hz
print("Grouping label dataframes")

Label_frames = [
    Label_data_df1,
    Label_data_df2,
    Label_data_df3,
    Label_data_df4,
    Label_data_df5,
    Label_data_df6,
    Label_data_df7,
    Label_data_df8,
    Label_data_df9,
    Label_data_df10,
    Label_data_df11,
    Label_data_df12,
    Label_data_df13
]

# Concatenate EEG dataframes into a single EEG dataframe and then delete the GSR data
print("Conatenating EEG dataframes")

EEG_df = pd.concat(EEG_frames)
del EEG_df[8]

# Concatenate training EEG dataframes into a single EEG dataframe and recode the values
print("Conatenating label dataframes")

Label_df = pd.concat(Label_frames)
Label_df = code_labels(Label_df)
Label_df = stringify_classes(Label_df)

# Add labels to a master df
print("Generating Mainframe")

Mainframe = EEG_df
Mainframe[8] = Label_df
Mainframe = Mainframe[Mainframe[8] != 4]
Mainframe = Mainframe[Mainframe[8] != 0] 
Mainframe = Mainframe[Mainframe[8] != 2]
Mainframe.reset_index(drop=True, inplace=True)

# Chunk the data for frequency analysis
print("Chunking for frequency analysis")

CleanChunkFrame = chunk_df(Mainframe, 1024)

# Make a dataframe of band power values averaged across all channels
print("Calculating average band power")

avg_bands_df = get_avg_band_df(CleanChunkFrame, eeg_channels, sampling_rate)

# Make a dataframe of band power values averaged across all channels
print("Calculating band power per chan")

bands_df = get_bands_df(CleanChunkFrame, eeg_channels, sampling_rate)

###########################################################################
###########################################################################

# MACHINE LEARNING STUFF ##################################################
###########################################################################

# Initialise Training Variables

# Raw
X_raw = Mainframe[[0, 1, 2, 3, 4, 5, 6, 7]]
y_raw = Mainframe[8]

# Averaged Bands
X_avg_bands = avg_bands_df[[0, 1, 2, 3, 4]]
y_avg_bands = avg_bands_df[5]

# Bands
X_bands = bands_df.iloc[:, 0:-1]
y_bands = bands_df.iloc[:,-1]

X = X_bands
y = y_bands

# Split data into a traning set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.10)

print("Fitting data to model... this may take a while")
classifier = svm.SVC(verbose=True, class_weight= 'balanced').fit(X_train, y_train)

np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
print("Plotting data to confusion matrix")

titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Blues, normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

print("Making final fit")
classifier = svm.SVC(verbose=True, class_weight= 'balanced').fit(X, y)

print("Saving classifier to disk")

# for saving the classifier
# s = pickle.dump(classifier, open('EmoSVM_TS.sav', 'wb'))
# s = pickle.dump(classifier, open('EmoSVM_Bands_avg3.sav', 'wb'))
s = pickle.dump(classifier, open('EmoSVM_Bands.sav', 'wb'))

