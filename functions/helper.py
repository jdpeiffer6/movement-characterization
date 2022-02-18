import pandas as pd
import os
from scipy import signal
import numpy as np

def readTaskCSV(path: str) -> pd.DataFrame:
    """Takes a path to specific session folder (marker or markerless) and returns a dataframe of the information"""
    if os.path.exists(path+'\\tasks.csv'):
        df = pd.read_csv(path+'\\tasks.csv')
        df.index=df['QTM Index']
        return df
    else:
        print("Path to task csv not found")
        return None

def getIndexNames(name: str, data: pd.DataFrame) -> list[str]:
    """Takes in a dataframe with Data Quality and Tasks names. Outputs the task names based on name argument"""
    data = data[data['Data Quality'] != 'Bad']
    return data[data['Task Name'] == name]['QTM Index']

def getMarkerlessData(markerlessData, key, features: list):
    data = markerlessData[key]
    data = data.loc[:,features]
    data = data.dropna(how='all')

    #check if there are any big jumps
    step_lengths = np.diff(data.index)
    if np.any(step_lengths!=1):
        jumps = np.where(step_lengths!=1)[0]
        if jumps.shape[0] > 1:
            print("Multiple discontunities detected...")
        data = data.iloc[(jumps[0]+1):,:]

    return data

def mapPeaks(peaks,interpx,xdata):
    locs = []
    peaks=peaks[0]
    for i in peaks:
        xval = interpx[i]    # this is what the interpolated value is
        
        #find where this is in the normal x data
        diffs = np.absolute(xdata-xval)
        locs.append(diffs.argmin())
    return np.sort(np.array(locs))   # note this returns peaks with reference to .values argument...its not absolute