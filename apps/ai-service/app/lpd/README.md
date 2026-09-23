# YOLO plate detection

See [the model workflow](../../../../docs/model-workflow.md) for training,
configuration, evaluation and deployment.

`PlateDetectorModel` loads trained YOLO weights once, predicts source-image boxes,
and returns `PlateBox` values. It never writes files. The shared frame processor
rounds boxes outward, clamps them and extracts original-resolution crops.
