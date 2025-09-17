# LAVA

LAVA (Latency Adaptive Video Analytics) is a framework that enforces strict latency bounds while maintaining the inference accuracy through fine-grained encoding parameter adaptation.

## Environmental Instructions:

The `LAVA_environment.yml` file defines a Conda environment named `LAVA`. This environment is built for Linux-64 systems that support GPU (CUDA).
It includes the following key components:
1. Python: 3.8.18
2. PyTorch: 2.1.2 (with CUDA 12 support via nvidia-cuda* and nvidia-cudnn-cu12 packages)
3. TorchVision: 0.16.2
4. Ultralytics YOLOv8: 8.2.52
5. OpenCV: 4.9.0.80 (via opencv-python) and base lib 4.6.0

## Video Size Predictor:
Contains pre trained model.

## Inference Accuracy Predictor:
Contains pre trained model.

## Video Sources:

Autonomous Driving (AD): https://www.youtube.com/watch?v=dIHYeTVklu4

Industrial Safety (IS): https://www.youtube.com/watch?v=vOG1Sm58NFc and https://www.youtube.com/watch?v=lfoTLeFooR4

Traffic Congestion (TC): https://www.youtube.com/watch?v=59c6yIYIys8

Street Facility (SF): https://drive.google.com/drive/folders/1SkogSHAO80lrMThiznVMfpUlbmsua8qR?usp=sharing
