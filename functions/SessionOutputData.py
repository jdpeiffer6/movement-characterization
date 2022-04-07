import pandas as pd
import numpy as np

class SessionOutputData:
    def __init__(self,id : str) -> None:
        self.id = id
        self.data : dict[str:dict[str:list[np.ndarray]]] = {}
    
    def addData(self, task_name : str,metric : str, data : np.ndarray):

        if task_name not in self.data.keys():
            self.data.update({task_name:dict()})
        if metric not in self.data[task_name].keys():
            self.data[task_name].update({metric:list()})
        
        self.data[task_name][metric].append(data)