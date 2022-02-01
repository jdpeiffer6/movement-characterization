# %%
import os
import DataObject
import matplotlib.pyplot as plt

from DataObject import DataObject
# path_to_output = "C:\\Users\\jd\\Documents\\movement-characterization\\data\\output"
path_to_output = "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output"
subjects = os.listdir(path_to_output)

#define your subject here
subject = '002'

subject_idx = subjects.index(subject)
path = path_to_output + '\\'+ subject


# %%
test_obj = DataObject(path)
# %%
