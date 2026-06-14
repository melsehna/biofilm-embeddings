"""Defaults for DINOv2 extraction. Most user-tunable values now live in
gui.state.AppState; this module just holds constants the model expects."""

# ImageNet normalization (DINOv2 was pretrained with these exact stats).
imagenetMean = [0.485, 0.456, 0.406]
imagenetStd  = [0.229, 0.224, 0.225]

# Fallback range used only when an input is *not* already in [0, 1] —
# matches the legacy phenotypr output range so that old _processed.tif
# files can still be fed in.
legacyDataRangeMin = -0.087
legacyDataRangeMax =  0.309
