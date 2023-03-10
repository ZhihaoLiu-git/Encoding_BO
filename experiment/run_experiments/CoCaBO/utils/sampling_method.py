import numpy as np

def generateInitialPoints(self, initN, bounds):
    nDim = len(bounds)
    Xinit = np.zeros((initN, len(bounds)))
    for i in range(initN):
        Xinit[i, :] = np.array(
            [np.random.uniform(bounds[b]['domain'][0],
                               bounds[b]['domain'][1], 1)[0]
             for b in range(nDim)])
    return Xinit