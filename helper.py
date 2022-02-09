import pandas as pd
import os

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
