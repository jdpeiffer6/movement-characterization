# %% Import data
from functions.SessionDataObject import SessionDataObject
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# path_to_subject2 =  "C:\\Users\\jpeiffer\\Documents\\cmdata\\004\\2022-04-12"
# subject2 = SessionDataObject(path_to_subject2,False,1.7,walking=True,ng=False)

# path_to_subject2_post =  "C:\\Users\\jpeiffer\\Documents\\cmdata\\006\\2022-06-20"
# subject2_post = SessionDataObject(path_to_subject2_post,False,1.7,walking=False,ng=False)

path_to_subject2_3mo =  "C:\\Users\\jpeiffer\\Documents\\cmdata\\007\\2022-08-26"
subject2_3mo = SessionDataObject(path_to_subject2_3mo,False,1.7,walking=False,ng=False)

# path_to_miguel = "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\002\\2022-01-28"
# miguel = SessionDataObject(path_to_miguel,False,1.75,walking=False,ng=False)
# labels=['#1BD9DE','#FE8821','#FEA832','#d8b365','#5ab4ac','#d8b365']