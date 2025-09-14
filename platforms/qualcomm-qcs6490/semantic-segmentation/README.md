# Semantic Segmentation Benchmark - Qualcomm QCS6490

This directory contains the complete benchmarking setup for Semantic Segmentation using DDRNet-23-slim on Qualcomm QCS6490 platforms.

## Overview

DDRNet-23-slim is optimized for efficient inference on Qualcomm's Hexagon DSP and Adreno GPU using the Qualcomm Neural Processing SDK (SNPE). This benchmark evaluates multi-accelerator performance on the Cityscapes dataset.

### Benchmark Objectives

- **Primary Metric**: Segmentation throughput (FPS)
- **Secondary Metrics**: DSP/GPU/CPU utilization, power consumption
- **Accuracy Metric**: Mean Intersection over Union (mIoU)
- **Platform Focus**: SNPE optimization across CPU, GPU, and DSP

## Prerequisites

### Hardware Requirements
- **Qualcomm QCS6490** (or Robotics RB5 development kit)
- **Active Cooling**: Heatsink or fan (recommended for sustained performance)
- **Storage**: 64GB+ eUFS or high-speed SD card
- **Power Supply**: 12V/3A power adapter
- **Optional**: Power measurement equipment

### Software Requirements
- **Ubuntu 20.04 LTS** (or compatible Linux distribution)
- **Qualcomm Neural Processing SDK (SNPE)** v2.x+
- **Adreno GPU drivers** (latest version)
- **OpenCV 4.5+** with optimization flags
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
- **Model Variants**: FP32 and quantized INT8 versions
- **Target Runtimes**: CPU, GPU (Adreno), DSP (Hexagon)
- **Input Resolutions**: 512x1024, 768x1536
- **Batch Sizes**: 1 (optimized for real-time inference)
- **Test Samples**: Cityscapes validation set (500 images)

### SNPE Optimizations
The benchmark automatically applies:
- **Quantization**: INT8 Post-Training Quantization for DSP
- **Graph Optimization**: Layer fusion and memory optimization
- **Runtime Selection**: Automatic fallback (DSP → GPU → CPU)
- **Buffer Management**: Optimized memory allocation
- **Multi-threading**: CPU runtime optimization

## Understanding Results

### Output Files
- `benchmark_results.json`: Performance metrics across all runtimes
- `system_info.json`: Hardware and software configuration
- `snpe_models/`: Converted SNPE DLC files
- `segmentation_results/`: Output segmentation masks
- `runtime_analysis.json`: Per-runtime performance breakdown
- `accuracy_analysis.json`: mIoU results for each runtime

### Key Metrics
- **Inference FPS**: Raw model inference throughput
- **End-to-End FPS**: Including pre/post-processing
- **Mean IoU**: Overall segmentation accuracy
- **Power Efficiency**: FPS per Watt
- **Runtime Utilization**: DSP/GPU/CPU usage percentages
- **Memory Usage**: Peak system and accelerator memory

### Runtime Performance Characteristics
1. **DSP (Hexagon)**: Highest efficiency, INT8 only
2. **GPU (Adreno)**: Balanced performance/power, FP16 support
3. **CPU (Kryo)**: Fallback option, full precision support

### Expected Performance Ranges

#### QCS6490 (DSP Runtime - INT8)
- **512x1024 Resolution**: 20-30 FPS
- **768x1536 Resolution**: 10-15 FPS
- **Power Consumption**: 8-12W
- **Memory Usage**: 1-2 GB system
- **DSP Utilization**: 70-90%

#### QCS6490 (GPU Runtime - FP16)
- **512x1024 Resolution**: 15-25 FPS
- **768x1536 Resolution**: 8-12 FPS
- **Power Consumption**: 10-15W
- **Memory Usage**: 1.5-2.5 GB system
- **GPU Utilization**: 60-80%

#### QCS6490 (CPU Runtime - FP32)
- **512x1024 Resolution**: 3-6 FPS
- **768x1536 Resolution**: 1.5-3 FPS
- **Power Consumption**: 6-10W
- **Memory Usage**: 1-1.5 GB system
- **CPU Utilization**: 80-95%

## Configuration Options

### Environment Variables
```bash
# Customize benchmark behavior
export SNPE_ROOT="/opt/qcom/aistack/snpe"
export SEGMENTATION_RUNTIME="dsp"       # Runtime: cpu, gpu, dsp
export SEGMENTATION_PRECISION="int8"    # Precision for quantized models
export SEGMENTATION_RESOLUTION="512x1024"  # Input resolution
export ENABLE_ACCURACY_EVAL=1           # Enable mIoU calculation
```

### Runtime Configuration
Edit the configuration in `run_benchmark.sh`:
```bash
# Runtimes to test
RUNTIMES=("dsp" "gpu" "cpu")

# Precision modes
PRECISIONS=("fp32" "int8")

# Input resolutions
RESOLUTIONS=("512x1024" "768x1536")
```

## Troubleshooting

### Common Issues

#### SNPE Runtime Errors
```bash
# Check SNPE installation
echo $SNPE_ROOT
ls $SNPE_ROOT/lib/

# Verify runtime availability
snpe-platform-validator

# Check DSP runtime
snpe-platform-validator --runtime dsp
```

#### Model Conversion Failures
```bash
# Rebuild SNPE models
rm -rf snpe_models/
REBUILD_MODELS=1 ./run_benchmark.sh

# Check conversion logs
cat logs/snpe_conversion.log
```

#### Low DSP Performance
```bash
# Check DSP firmware
cat /sys/kernel/debug/msm_subsys/slpi

# Verify DSP clocks
cat /sys/kernel/debug/clk/clk_summary | grep dsp
```

#### Memory Issues
```bash
# Monitor memory usage
free -h
cat /proc/meminfo

# Check for memory leaks
valgrind --leak-check=full python3 benchmark_script.py
```

### Performance Debugging
```bash
# Run with SNPE profiling
ENABLE_SNPE_PROFILING=1 ./run_benchmark.sh

# Monitor system resources
htop  # In another terminal
```

## Scientific Methodology

### Measurement Protocol
1. **System Stabilization**: 5-minute idle period
2. **Runtime Warm-up**: 50 inference iterations per runtime
3. **Measurement Phase**: 500+ samples with detailed timing
4. **Accuracy Evaluation**: Full validation set processing

### Quantization Process
- **Calibration Dataset**: 200 representative Cityscapes images
- **Quantization Method**: SNPE INT8 Post-Training Quantization
- **Validation**: mIoU comparison across runtimes
- **Fallback Handling**: Automatic runtime fallback on errors

### Statistical Analysis
- Per-runtime latency distributions
- Cross-runtime performance comparison
- Power efficiency analysis
- Thermal impact on sustained performance

## Advanced Usage

### Custom Models
```bash
# Use custom DDRNet ONNX model
export DDRNET_ONNX_PATH="/path/to/custom_ddrnet.onnx"
./run_benchmark.sh
```

### Multi-Runtime Comparison
```bash
# Compare all runtimes
python3 ../../../analysis/scripts/compare_snpe_runtimes.py \
    results/run_*_dsp/ \
    results/run_*_gpu/ \
    results/run_*_cpu/
```

### Power-Performance Analysis
```bash
# Enable power measurement
export ENABLE_POWER_MEASUREMENT=1
./run_benchmark.sh

# Analyze efficiency across runtimes
python3 ../../../analysis/scripts/snpe_efficiency_analysis.py results/
```

### Custom Dataset Evaluation
```bash
# Run on custom dataset
export CUSTOM_DATASET_PATH="/path/to/images"
export CUSTOM_LABELS_PATH="/path/to/labels"
./run_benchmark.sh --custom-dataset
```

## Model Details

### DDRNet-23-slim for SNPE
- **Architecture**: Dual-resolution dual-branch network
- **SNPE Compatibility**: Optimized for Hexagon DSP
- **Input Size**: Flexible (512x1024, 768x1536 tested)
- **Quantization**: INT8 for DSP, FP16 for GPU
- **Output**: Per-pixel class predictions (19 classes)

### SNPE Optimizations
- **Graph Optimization**: Layer fusion and redundancy removal
- **Memory Layout**: Optimized tensor formats for each runtime
- **Precision Handling**: Mixed precision where supported
- **Batch Processing**: Optimized for batch size 1

## References

- [DDRNet Paper](https://arxiv.org/abs/2101.06085)
- [Qualcomm SNPE Documentation](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [QCS6490 Technical Reference](https://www.qualcomm.com/products/internet-of-things/industrial/building-enterprise/qcs6490)

## Support

For issues specific to this benchmark:
1. Check the troubleshooting section above
2. Review SNPE installation in `../setup/README.md`
3. Consult Qualcomm Developer documentation
4. Verify runtime availability with `snpe-platform-validator`
