# Semantic Segmentation Benchmark - Radxa CM5

This directory contains the complete benchmarking setup for Semantic Segmentation using DDRNet-23-slim on Radxa CM5 platforms with RK3588S processor.

## Overview

DDRNet-23-slim is optimized for efficient inference on RK3588S using RKNN toolkit. This benchmark evaluates NPU, Mali GPU, and ARM CPU performance on the Cityscapes dataset.

### Benchmark Objectives

- **Primary Metric**: Segmentation throughput (FPS)
- **Secondary Metrics**: NPU/GPU/CPU utilization, power consumption
- **Accuracy Metric**: Mean Intersection over Union (mIoU)
- **Platform Focus**: RKNN optimization for RK3588S NPU, Mali GPU, and ARM CPU

## Prerequisites

### Hardware Requirements
- **Radxa CM5** with RK3588S processor
- **Active Cooling**: Heatsink or fan (recommended for sustained performance)
- **Storage**: 64GB+ eMMC or high-speed microSD card (Class 10+)
- **Power Supply**: 12V/2A DC or USB-C PD (24W minimum)
- **Optional**: Power measurement equipment

### Software Requirements
- **Ubuntu 20.04 LTS** (ARM64) or Debian 11
- **RKNN Toolkit** 1.5.x+
- **Mali GPU Drivers** (latest version)
- **OpenCV 4.5+** with ARM optimizations
- **Python 3.8+** with NumPy, Pillow

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
- **Target Devices**: CPU (Intel N100), GPU (Intel UHD Graphics)
- **Input Resolutions**: 512x1024, 768x1536
- **Batch Sizes**: 1, 2, 4 (memory permitting)
- **Test Samples**: Cityscapes validation set (500 images)

### OpenVINO Optimizations
The benchmark automatically applies:
- **Model Optimization**: OpenVINO Model Optimizer (MO)
- **Quantization**: Post-training Optimization Tool (POT) for INT8
- **Runtime Optimization**: Inference Engine optimization
- **Threading**: Multi-threading for CPU inference
- **Memory Management**: Optimized memory allocation

## Understanding Results

### Output Files
- `benchmark_results.json`: Performance metrics across CPU/GPU
- `system_info.json`: Hardware and software configuration
- `openvino_models/`: Converted OpenVINO IR files
- `segmentation_results/`: Output segmentation masks
- `device_analysis.json`: Per-device performance breakdown
- `accuracy_analysis.json`: mIoU results for each device

### Key Metrics
- **Inference FPS**: Raw model inference throughput
- **End-to-End FPS**: Including pre/post-processing
- **Mean IoU**: Overall segmentation accuracy
- **Power Efficiency**: FPS per Watt (if power measurement available)
- **Device Utilization**: CPU/GPU usage percentages
- **Memory Usage**: Peak system and device memory

### Device Performance Characteristics
1. **CPU (Intel N100)**: Multi-threaded, supports all precisions
2. **GPU (Intel UHD)**: Parallel processing, optimized for FP16/INT8

### Expected Performance Ranges

#### Intel N100 CPU (FP32)
- **512x1024 Resolution**: 8-12 FPS
- **768x1536 Resolution**: 4-7 FPS
- **Power Consumption**: 6-10W
- **Memory Usage**: 1-2 GB system
- **CPU Utilization**: 70-90%

#### Intel N100 CPU (INT8)
- **512x1024 Resolution**: 15-25 FPS
- **768x1536 Resolution**: 8-12 FPS
- **Power Consumption**: 8-12W
- **Memory Usage**: 1-1.5 GB system
- **CPU Utilization**: 80-95%

#### Intel UHD Graphics (FP16)
- **512x1024 Resolution**: 12-18 FPS
- **768x1536 Resolution**: 6-10 FPS
- **Power Consumption**: 8-15W
- **Memory Usage**: 1.5-2.5 GB system
- **GPU Utilization**: 60-80%

## Configuration Options

### Environment Variables
```bash
# Customize benchmark behavior
export INTEL_OPENVINO_DIR="/opt/intel/openvino"
export SEGMENTATION_DEVICE="CPU"         # Device: CPU, GPU
export SEGMENTATION_PRECISION="int8"     # Precision mode
export SEGMENTATION_RESOLUTION="512x1024"  # Input resolution
export OMP_NUM_THREADS=4                 # CPU threading
export ENABLE_ACCURACY_EVAL=1            # Enable mIoU calculation
```

### Device Configuration
Edit the configuration in `run_benchmark.sh`:
```bash
# Devices to test
DEVICES=("CPU" "GPU")

# Precision modes
PRECISIONS=("fp32" "fp16" "int8")

# Input resolutions
RESOLUTIONS=("512x1024" "768x1536")

# Batch sizes
BATCH_SIZES=(1 2 4)
```

## Troubleshooting

### Common Issues

#### OpenVINO Runtime Errors
```bash
# Check OpenVINO installation
echo $INTEL_OPENVINO_DIR
ls $INTEL_OPENVINO_DIR/runtime/lib/intel64/

# Verify device availability
python3 -c "from openvino.runtime import Core; print(Core().available_devices)"
```

#### Model Conversion Failures
```bash
# Rebuild OpenVINO models
rm -rf openvino_models/
REBUILD_MODELS=1 ./run_benchmark.sh

# Check conversion logs
cat logs/openvino_conversion.log
```

#### GPU Runtime Issues
```bash
# Check Intel GPU driver
lsmod | grep i915

# Verify GPU device
ls /dev/dri/
intel_gpu_top  # If available
```

#### Low Performance
```bash
# Check CPU governor
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Set performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Check CPU frequencies
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
```

### Performance Debugging
```bash
# Run with OpenVINO profiling
ENABLE_OPENVINO_PROFILING=1 ./run_benchmark.sh

# Monitor system resources
htop  # In another terminal
```

## Scientific Methodology

### Measurement Protocol
1. **System Stabilization**: 5-minute idle period
2. **Runtime Warm-up**: 50 inference iterations per device
3. **Measurement Phase**: 500+ samples with detailed timing
4. **Accuracy Evaluation**: Full validation set processing

### Quantization Process
- **Calibration Dataset**: 200 representative Cityscapes images
- **Quantization Tool**: OpenVINO Post-training Optimization Tool (POT)
- **Validation**: mIoU comparison across precisions
- **Optimization**: Device-specific optimization settings

### Statistical Analysis
- Per-device latency distributions
- Cross-device performance comparison
- Power efficiency analysis (if power measurement available)
- Thermal impact on sustained performance

## Advanced Usage

### Custom Models
```bash
# Use custom DDRNet ONNX model
export DDRNET_ONNX_PATH="/path/to/custom_ddrnet.onnx"
./run_benchmark.sh
```

### Multi-Device Comparison
```bash
# Compare CPU vs GPU performance
python3 ../../../analysis/scripts/compare_openvino_devices.py \
    results/run_*_cpu/ \
    results/run_*_gpu/
```

### Power-Performance Analysis
```bash
# Enable power measurement (if available)
export ENABLE_POWER_MEASUREMENT=1
./run_benchmark.sh

# Analyze efficiency
python3 ../../../analysis/scripts/intel_efficiency_analysis.py results/
```

### Custom Dataset Evaluation
```bash
# Run on custom dataset
export CUSTOM_DATASET_PATH="/path/to/images"
export CUSTOM_LABELS_PATH="/path/to/labels"
./run_benchmark.sh --custom-dataset
```

## Model Details

### DDRNet-23-slim for OpenVINO
- **Architecture**: Dual-resolution dual-branch network
- **OpenVINO Compatibility**: Optimized for Intel hardware
- **Input Size**: Flexible (512x1024, 768x1536 tested)
- **Quantization**: INT8 for CPU, FP16 for GPU
- **Output**: Per-pixel class predictions (19 classes)

### OpenVINO Optimizations
- **Graph Optimization**: Layer fusion and constant folding
- **Memory Layout**: Optimized tensor formats for Intel devices
- **Precision Handling**: Mixed precision where supported
- **Threading**: OpenMP optimization for CPU inference

## References

- [DDRNet Paper](https://arxiv.org/abs/2101.06085)
- [Intel OpenVINO Documentation](https://docs.openvino.ai/)
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [Intel N100 Technical Specifications](https://ark.intel.com/content/www/us/en/ark/products/231803/intel-processor-n100-6m-cache-up-to-3-40-ghz.html)

## Support

For issues specific to this benchmark:
1. Check the troubleshooting section above
2. Review OpenVINO installation in `../setup/README.md`
3. Consult Intel OpenVINO documentation
4. Verify device availability with OpenVINO runtime
