import numpy as np
import cv2

def computeMaskInplace(stack, masks, fixedThresh):
    masks[:] = stack > float(fixedThresh)
    return masks

def dustCorrectInplace(masks):
    firstOn = masks[..., 0]
    everOff = np.any(~masks[..., 1:], axis=2)
    kill = firstOn & everOff
    masks[kill, :] = False
    return masks
