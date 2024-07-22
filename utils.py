import os
import librosa
import numpy as np 
from datetime import datetime, timezone


#####################
# Functions section #
#####################################
# geting the features from the voice
#####################################

## ZCR: Zero Crossing Rate: The rate of sign changes of the signal during the duration of a particular frame
def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

## RMS: root mean square value
def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

## MFCC: Mel Frequency Cepstral Coefficients form a cepstral representation where the frequency bands are not linear but distributed according to the mel-scale
def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    mfcc=librosa.feature.mfcc(y=data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

## Extraxing the features
def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])

    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                        ))
    return result

def get_date_string():
    current_datetime = datetime.now(timezone.utc)

    # Format the datetime as desired
    formatted_datetime = current_datetime.strftime('%Y_%m_%d_%H-%M')
    return formatted_datetime



def get_latest_experiment(experiments_dir = r"tmp"):

    # List all folders in the experiments directory
    experiment_folders = [folder for folder in os.listdir(experiments_dir)]
    if not experiment_folders: #check if empty
        return "exp_" + get_date_string()
    
    # Parse folder names and extract datetime information
    parsed_folders = []
    for folder_name in experiment_folders:
        try:
            folder_datetime = datetime.strptime(folder_name, 'exp_%Y_%m_%d_%H-%M')
            parsed_folders.append((folder_datetime, folder_name))
        except ValueError:
            print(ValueError)
            # Skip folders with names not matching the expected format
            pass

    # Sort the parsed folders based on datetime
    sorted_folders = sorted(parsed_folders, key=lambda x: x[0], reverse=True)

    # Retrieve the latest folder name
    latest_folder = sorted_folders[0][1] if sorted_folders else None

    print("Latest experiment folder:", latest_folder)

    return latest_folder
