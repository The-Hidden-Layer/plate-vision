# Iranian plate recognition

See [the model workflow](../../../../docs/model-workflow.md) for training,
configuration, evaluation and deployment.

`PlateRecognizerModel` loads a versioned LPRNet checkpoint or TensorRT engine.
It supports CUDA, native MPS and CPU. `read_batch()` recognizes the crops within
one frame, preserving slot order; `read()` remains available for callers. Empty
text means unreadable; the shared pipeline retains that crop.

`alphabet.py` owns normalization/tokenization, `network.py` owns the RGB 192×48
preprocessing and fully convolutional CTC network. Models never write files.
