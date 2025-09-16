# Pre-trained ONNX Models

This directory contains pre-trained ONNX models for the benchmark tasks, hosted directly in the repository for faster downloads across all platforms.

## Available Models

### 3D Object Detection
- **`crestereo.onnx`** (25MB) - CREStereo stereo depth estimation model (480x640 resolution)
  - Source: PINTO0309 model zoo
  - License: Apache-2.0
  - Original: [CREStereo](https://github.com/megvii-research/CREStereo)

- **`pointpillars.onnx`** (18MB) - PointPillars 3D object detection model
  - Source: NVIDIA-AI-IOT CUDA-PointPillars
  - License: Apache-2.0
  - Original: [NVIDIA PointPillars](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars)

### Semantic Segmentation
- **`ddrnet.onnx`** (TBD) - DDRNet semantic segmentation model
  - Will be added when needed

### ORB-SLAM3
- ORB-SLAM3 uses its own vocabulary and configuration files (not ONNX models)

## Usage

These models are automatically downloaded by the unified download scripts:
- `../datasets/download_all_models.py` - Downloads models for specific benchmarks
- `../datasets/download_all_datasets.py` - Downloads both datasets and models

## Size Comparison

| Model | Size | Download Time (Est.) |
|-------|------|---------------------|
| CREStereo | 25MB | ~5-10 seconds |
| PointPillars | 18MB | ~3-7 seconds |
| **Total** | **43MB** | **~10-20 seconds** |

*Much faster than downloading 264MB+ archives and extracting!*

## License

Models are provided under their respective licenses:
- CREStereo: Apache-2.0 License
- PointPillars: Apache-2.0 License

See individual model repositories for detailed license information.