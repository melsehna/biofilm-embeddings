import numpy as np
from numpy import pi

def roundOdd(x):
    x = int(round(x))
    return x if x % 2 else x + 1

def compmax(x):
    return np.max(x) if len(x) > 1 else 0

def calculateStats(crosscorMaxima, sourceFreq, targetFreq):
    sourceAmp = np.mean(np.abs(sourceFreq) ** 2)
    targetAmp = np.mean(np.abs(targetFreq) ** 2)
    error = 1 - (np.abs(crosscorMaxima) ** 2) / (sourceAmp * targetAmp)
    phasediff = np.angle(crosscorMaxima)
    return error, phasediff
