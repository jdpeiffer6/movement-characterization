import numpy as np
import math

def rms(in_vec : np.ndarray):
    if len(in_vec.shape) == 1:
        n = in_vec.shape
        square = 0.0
        root = 0.0
        mean = 0.0

        for i in range(n[0]):
            square += in_vec[i]**2
        mean = (square / (float)(n[0]))

        root = math.sqrt(mean)
        
        return root
    else:
        print("Wrong array dimensions")
        return -1
