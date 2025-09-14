# Semantic Segmentation Benchmark - NVIDIA Jetson

This directory contains the complete benchmarking setup for Semantic Segmentation using DDRNet-23-slim on NVIDIA Jetson platforms.

## Overview

DDRNet-23-slim is a lightweight, real-time semantic segmentation network designed for efficient inference on edge devices. This benchmark evaluates both CPU and GPU performance using TensorRT optimization on the Cityscapes dataset.

### Benchmark Objectives

- **Primary Metric**: Segmentation throughput (FPS)
- **Secondary Metrics**: GPU/CPU utilization, memory usage, power consumption
- **Accuracy Metric**: Mean Intersection over Union (mIoU)
- **Platform Focus**: Real-time inference optimization with TensorRT

## Prerequisites

### Hardware Requirements
- NVIDIA Jetson Orin NX (8GB or 16GB)
- Active cooling solution (recommended for sustained performance)
- MicroSD card (64GB+, Class 10 or better)
- Power measurement equipment (optional but recommended)

### Software Requirements
- Ubuntu 20.04 LTS (JetPack 5.1.1)
- CUDA 11.4+ with cuDNN
- TensorRT 8.5.2+
- OpenCV 4.5+ with CUDA support
- PyTorch 1.13+ with CUDA support
- ONNX and onnx-tensorrt

### Dataset Requirements
- Cityscapes dataset (validation set)
- Approximately 2GB storage space

## Quick Start

1. **Ensure prerequisites are installed:**
   ```bash
   cd ../setup
   ./install_all.sh
   ```

2. **Run the benchmark:**
   ```bash
   ./run_benchmark.sh
   ```

3. **View results:**
   ```bash
   ls results/
   # Check the latest timestamped results folder
   ```

## Benchmark Configuration

### Default Settings
- **Model Variants**: FP32, FP16, and INT8 quantized versions
- **Input Resolutions**: 512x1024, 768x1536 (Cityscapes standard)
- **Batch Sizes**: 1, 2, 4 (memory permitting)
- **Inference Engines**: TensorRT optimized
- **Test Samples**: Cityscapes validation set (500 images)

### TensorRT Optimizations
The benchmark automatically applies:
- Layer fusion and kernel optimization
- FP16 and INT8 quantization
- Dynamic shape optimization
- Memory pool management
- CUDA stream optimization

## Understanding Results

### Output Files
- `benchmark_results.json`: Complete performance and accuracy metrics
- `system_info.json`: Hardware and software configuration
- `tensorrt_engines/`: Optimized TensorRT engines
- `segmentation_results/`: Output segmentation masks
- `timing_analysis.json`: Detailed inference timing
- `accuracy_analysis.json`: Per-class mIoU results

### Key Metrics
- **Inference FPS**: Raw model inference throughput
- **End-to-End FPS**: Including pre/post-processing
- **Mean IoU**: Overall segmentation accuracy
- **Per-Class IoU**: Individual class performance
- **GPU Utilization**: Average GPU usage percentage
- **Memory Usage**: Peak GPU and system memory

### Processing Pipeline
1. **Image Preprocessing**: ~5-10% of total time
2. **Model Inference**: ~70-80% of total time
3. **Post-processing**: ~10-15% of total time
4. **Visualization/Saving**: ~5-10% of total time

### Expected Performance Ranges

#### Jetson Orin NX 16GB
- **FP32 Inference**: 25-35 FPS (512x1024), 15-20 FPS (768x1536)
- **FP16 Inference**: 40-55 FPS (512x1024), 20-30 FPS (768x1536)
- **INT8 Inference**: 60-80 FPS (512x1024), 30-45 FPS (768x1536)
- **GPU Utilization**: 70-90%
- **Memory Usage**: 1-3 GB GPU, 1-2 GB system
- **Power**: 15-25W during inference

#### Jetson Orin NX 8GB
- **FP32 Inference**: 20-30 FPS (512x1024), 12-18 FPS (768x1536)
- **FP16 Inference**: 35-50 FPS (512x1024), 18-25 FPS (768x1536)
- **INT8 Inference**: 55-75 FPS (512x1024), 25-40 FPS (768x1536)
- **GPU Utilization**: 75-95%
- **Memory Usage**: 1-2.5 GB GPU, 1-1.5 GB system
- **Power**: 12-22W during inference

## Configuration Options

### Environment Variables
```bash
# Customize benchmark behavior
export SEGMENTATION_BATCH_SIZE=2        # Batch size for inference
export SEGMENTATION_PRECISION="fp16"    # Precision mode
export SEGMENTATION_RESOLUTION="512x1024"  # Input resolution
export TENSORRT_WORKSPACE_SIZE=2048     # TensorRT workspace in MB
export ENABLE_ACCURACY_EVAL=1           # Enable mIoU calculation
```

### Model Configuration
Edit the configuration in `run_benchmark.sh`:
```bash
# Precision modes to test
PRECISIONS=("fp32" "fp16" "int8")

# Input resolutions
RESOLUTIONS=("512x1024" "768x1536")

# Batch sizes
BATCH_SIZES=(1 2 4)
```

## Troubleshooting

### Common Issues

#### GPU Memory Errors
- **Reduce batch size**: Set `SEGMENTATION_BATCH_SIZE=1`
- **Lower resolution**: Use `SEGMENTATION_RESOLUTION="512x1024"`
- **Check memory**: Monitor with `nvidia-smi`

#### TensorRT Build Failures
```bash
# Rebuild TensorRT engines
rm -rf tensorrt_engines/
REBUILD_ENGINES=1 ./run_benchmark.sh
```

#### Low Performance
- **Check thermal status**: Monitor GPU temperatures
- **Verify clocks**: Use `sudo jetson_clocks`
- **Check power mode**: Ensure MAXN power mode

#### Accuracy Degradation
```bash
# Verify quantization quality
python3 scripts/validate_quantization.py

# Check calibration data
ls calibration_data/
```

### Performance Debugging
```bash
# Run with detailed profiling
ENABLE_PROFILING=1 ./run_benchmark.sh

# Monitor system during inference
sudo tegrastats  # In another terminal
```

## Scientific Methodology

### Measurement Protocol
1. **System Stabilization**: 5-minute idle period
2. **Engine Warm-up**: 100 inference iterations
3. **Measurement Phase**: 500+ samples with timing
4. **Accuracy Evaluation**: Full validation set processing

### Quantization Calibration
- **Calibration Dataset**: 200 representative Cityscapes images
- **Calibration Method**: Entropy-based Post-Training Quantization
- **Validation**: mIoU comparison with FP32 baseline

### Statistical Analysis
- Latency distribution analysis (P50, P95, P99)
- Throughput confidence intervals
- Performance stability over time
- Thermal impact on performance

## Advanced Usage

### Custom Models
```bash
# Use custom DDRNet model
export DDRNET_MODEL_PATH="/path/to/custom_ddrnet.onnx"
./run_benchmark.sh
```

### Multi-Resolution Analysis
```bash
# Test multiple resolutions
for res in "512x1024" "768x1536" "1024x2048"; do
    SEGMENTATION_RESOLUTION=$res ./run_benchmark.sh
done
```

### Power-Performance Profiling
```bash
# Enable power measurement
export ENABLE_POWER_MEASUREMENT=1
export POWER_METER_IP="192.168.1.100"
./run_benchmark.sh

# Analyze efficiency
python3 ../../../analysis/scripts/efficiency_analysis.py results/
```

### Custom Dataset Evaluation
```bash
# Run on custom dataset
export CUSTOM_DATASET_PATH="/path/to/images"
export CUSTOM_LABELS_PATH="/path/to/labels"
./run_benchmark.sh --custom-dataset
```

## Model Details

### DDRNet-23-slim
- **Architecture**: Dual-resolution dual-branch network
- **Parameters**: ~5.7M (lightweight design)
- **Input Size**: Flexible (tested at 512x1024, 768x1536)
- **Classes**: 19 Cityscapes classes
- **Optimization**: TensorRT with mixed precision support

### Network Architecture
- **Backbone**: Efficient dual-branch design
- **Feature Fusion**: Deep aggregation between branches
- **Output**: Per-pixel class predictions
- **Post-processing**: Argmax and colormap application

## References

- [DDRNet Paper](https://arxiv.org/abs/2101.06085)
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/best-practices/)
- [Jetson Performance Tuning](https://docs.nvidia.com/jetson/archives/r35.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance.html)

## Support

For issues specific to this benchmark:
1. Check the troubleshooting section above
2. Review TensorRT setup in `../setup/README.md`
3. Consult the main project documentation in `../../../docs/`
4. Verify GPU compatibility and drivers with `nvidia-smi`
