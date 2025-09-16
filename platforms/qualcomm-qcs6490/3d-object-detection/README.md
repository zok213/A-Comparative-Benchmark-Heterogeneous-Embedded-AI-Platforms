# 3D Object Detection Benchmark - Qualcomm QCS6490

This directory contains the complete setup and execution scripts for benchmarking 3D Object Detection performance on the Qualcomm QCS6490 platform using the Pseudo-LiDAR + PointPillars pipeline.

## 📋 Overview

**Benchmark Type**: AI Accelerator Performance Evaluation  
**Target Platform**: Qualcomm QCS6490 (Hexagon DSP/NPU)  
**Workload**: Pseudo-LiDAR + PointPillars 3D Object Detection Pipeline  
**Dataset**: KITTI 3D Object Detection  
**Key Metrics**: End-to-end Latency, Throughput (FPS), 3D mAP, Power Consumption  

## 🎯 Benchmark Objective

This benchmark evaluates the AI accelerator performance of the Qualcomm QCS6490 platform using a two-stage 3D object detection pipeline that represents real-world autonomous driving perception workloads:

### Pipeline Architecture
1. **Stage 1**: Stereo depth estimation using CREStereo
2. **Intermediate**: Pseudo-LiDAR point cloud generation (CPU-based)
3. **Stage 2**: 3D object detection using PointPillars

### Why This Benchmark?
- **Real-world relevance**: Mimics autonomous vehicle perception pipelines
- **Heterogeneous compute**: Tests both AI accelerators and CPU
- **Memory intensive**: Stresses memory bandwidth and cache hierarchy
- **Quantization evaluation**: Tests INT8 performance on Hexagon NPU

## 🛠️ Prerequisites

### Hardware Requirements
- Qualcomm QCS6490 development board (Thundercomm TurboX C6490 recommended)
- Active cooling solution (40mm fan + heatsink) - **MANDATORY**
- Stable 12V/3A power supply
- microSD card (128GB+) or eUFS storage (dataset is ~12GB)
- Yokogawa WT300E power analyzer (for accurate power measurement)

### Software Requirements
- Ubuntu 20.04 LTS (ARM64)
- Qualcomm Neural Processing SDK (SNPE) v2.x+ properly installed
- Platform setup completed (run `../setup/install_all.sh` first)
- KITTI dataset (use optimized download script below)

## 🚀 Quick Start

### 1. Download Models and Datasets
```bash
# Option 1: Download both models and datasets together
python3 ../../../datasets/download_all_datasets.py --datasets kitti --include-models

# Option 2: Download separately with specific control
python3 ../../../datasets/download_all_models.py --benchmarks 3d-detection
python3 ../../../datasets/download_kitti_with_cookies.py
# Downloads: scene_flow + object_calib for stereo depth + 3D detection

# Models downloaded: CREStereo + PointPillars (ONNX format)
# Platform-specific optimization (SNPE DLC) happens automatically during benchmark
```

### 2. Verify Prerequisites
```bash
# Check if SNPE is properly installed
echo $SNPE_ROOT
ls $SNPE_ROOT/bin/x86_64-linux-clang/snpe-onnx-to-dlc

# Check if KITTI dataset is available
ls ~/benchmark_workspace/datasets/kitti/

# Check environment variables
source ~/benchmark_workspace/setup_env.sh
echo $MODELS_ROOT
echo $DATASETS_ROOT
```

### 2. Run Benchmark
```bash
cd qualcomm-qcs6490/3d-object-detection/
chmod +x run_benchmark.sh
./run_benchmark.sh
```

### 3. View Results
```bash
# View CPU results
cat ~/benchmark_workspace/results/3d_detection/3d_detection_results_cpu_*.json

# View DSP/NPU results  
cat ~/benchmark_workspace/results/3d_detection/3d_detection_results_dsp_*.json
```

## 📊 Understanding Results

### Output Files
```
~/benchmark_workspace/results/3d_detection/
├── 3d_detection_results_cpu_*.json    # CPU benchmark results
├── 3d_detection_results_dsp_*.json    # DSP/HTP benchmark results
├── logs/                              # Execution logs
│   ├── cpu_benchmark.log
│   └── dsp_benchmark.log
├── calibration/                       # Calibration data
│   ├── image_2/                       # Left stereo images
│   ├── image_3/                       # Right stereo images
│   ├── calib/                         # Camera calibration
│   ├── raw/                           # Raw binary files for SNPE
│   └── calibration_list.txt           # SNPE calibration file list
└── models/
    └── snpe/                          # Converted SNPE models
        ├── crestereo_cpu.dlc          # CREStereo for CPU
        ├── crestereo_dsp.dlc          # CREStereo for DSP/HTP
        ├── pointpillars_cpu.dlc       # PointPillars for CPU
        └── pointpillars_dsp.dlc       # PointPillars for DSP/HTP
```

### Key Metrics Explained

#### Performance Metrics
- **Stereo Latency (ms)**: Time for depth estimation from stereo images
- **Conversion Latency (ms)**: CPU time to generate pseudo-LiDAR point cloud
- **Detection Latency (ms)**: Time for 3D object detection from point cloud
- **Total Pipeline Latency (ms)**: End-to-end processing time
- **Throughput (FPS)**: Frames processed per second

#### Statistical Measures
- **Mean**: Average processing time across all iterations
- **P99**: 99th percentile latency (worst-case for 99% of frames)
- **P95/P90**: 95th/90th percentile latencies
- **Standard Deviation**: Consistency measure

#### Accelerator Comparison
- **CPU Runtime**: Uses Qualcomm Kryo 670 CPU cores
- **DSP/HTP Runtime**: Uses Hexagon Tensor Processor (NPU)

### Expected Results (Reference)
Based on Qualcomm QCS6490 specifications:

| Component | CPU Runtime | DSP/HTP Runtime | Notes |
|-----------|-------------|-----------------|-------|
| Stereo Estimation | 150-300 ms | 80-150 ms | NPU acceleration |
| Point Cloud Conv. | 15-30 ms | 15-30 ms | CPU-only operation |
| 3D Detection | 200-400 ms | 100-200 ms | NPU acceleration |
| **Total Pipeline** | **365-730 ms** | **195-380 ms** | **End-to-end** |
| **Throughput** | **1.4-2.7 FPS** | **2.6-5.1 FPS** | **Sustained** |
| Power Consumption | 8-12 W | 10-15 W | Higher for NPU |

## ⚙️ Configuration Options

### Benchmark Parameters
Modify these parameters in the script:

```bash
NUM_ITERATIONS=1000         # Number of benchmark iterations
WARMUP_ITERATIONS=100       # Warmup iterations before measurement
```

### Model Optimization
The benchmark supports different runtime targets:
- **CPU**: Uses ARM Cortex-A78AE cores
- **DSP/HTP**: Uses Hexagon Tensor Processor (NPU)

### Input Configurations
- **Image Resolution**: 1242x375 (KITTI standard)
- **Point Cloud Size**: Up to 60,000 points
- **Quantization**: INT8 for both models

## 🔧 Troubleshooting

### Common Issues

#### 1. SNPE SDK Not Found
```bash
# Error: SNPE SDK not found
# Solution: Install and source SNPE properly
export SNPE_ROOT=/path/to/snpe-sdk
source $SNPE_ROOT/bin/envsetup.sh
```

#### 2. Model Conversion Failures
```bash
# Check SNPE tools availability
ls $SNPE_ROOT/bin/x86_64-linux-clang/snpe-onnx-to-dlc

# Verify ONNX models exist
ls ~/benchmark_workspace/models/onnx/

# Manual model conversion
cd ~/benchmark_workspace/models/onnx/
$SNPE_ROOT/bin/x86_64-linux-clang/snpe-onnx-to-dlc \
    --input_network crestereo.onnx \
    --output_path ../snpe/crestereo_fp32.dlc
```

#### 3. KITTI Dataset Issues
```bash
# Verify dataset structure
tree ~/benchmark_workspace/datasets/kitti/object/ -L 3

# Re-download if corrupted
cd ../../datasets/
./prepare_all_datasets.sh
```

#### 4. Runtime Failures

**DSP/HTP Runtime Not Available**:
```bash
# Check available runtimes
$SNPE_ROOT/bin/aarch64-android/snpe-platform-validator

# Fall back to CPU if DSP unavailable
# Edit run_benchmark.sh and comment out DSP benchmark
```

**Memory Issues**:
```bash
# Reduce batch size or iterations
export NUM_ITERATIONS=100
export WARMUP_ITERATIONS=10

# Check available memory
free -h
```

**Performance Issues**:
- Ensure active cooling is working
- Check CPU frequencies: `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq`
- Verify power supply stability (12V/3A minimum)

### Debug Mode
Run with verbose output for troubleshooting:
```bash
bash -x run_benchmark.sh
```

### Manual Model Testing
Test individual models:
```bash
cd ~/benchmark_workspace/results/3d_detection/

# Test CREStereo model
$SNPE_ROOT/bin/aarch64-android/snpe-net-run \
    --container ../models/snpe/crestereo_dsp.dlc \
    --input_list calibration/calibration_list.txt \
    --use_dsp
```

## 🔬 Scientific Methodology

### Experimental Controls
- **Thermal Stability**: Active cooling prevents performance degradation
- **Model Consistency**: Identical ONNX models converted for each runtime
- **Dataset Consistency**: Same KITTI validation set for all tests
- **Statistical Rigor**: Multiple iterations with proper statistical analysis

### Quantization Methodology
- **Post-Training Quantization (PTQ)**: INT8 quantization using KITTI calibration data
- **Calibration Dataset**: 500 randomly selected KITTI training images
- **Accuracy Preservation**: Quantization parameters optimized for minimal accuracy loss

### Power Measurement
- **Hardware-based**: External Yokogawa WT300E power analyzer
- **System-level**: Measures total board power consumption
- **Synchronized**: Power logging aligned with benchmark execution

## 📖 Advanced Usage

### Custom Model Testing
```bash
# Test your own ONNX models
cp your_model.onnx ~/benchmark_workspace/models/onnx/
# Update model paths in run_benchmark.sh
```

### Profiling and Analysis
```bash
# Enable SNPE profiling
export SNPE_PROFILE=1

# Analyze layer-wise performance
$SNPE_ROOT/bin/x86_64-linux-clang/snpe-diagview \
    --input_log profile_data.log
```

### Accuracy Evaluation
```bash
# Run accuracy evaluation (requires ground truth)
cd ~/benchmark_workspace/results/3d_detection/
python3 evaluate_3d_detection.py \
    --predictions detections/ \
    --ground_truth ../../datasets/kitti/object/training/label_2/
```

## 📚 References

1. [Pseudo-LiDAR from Visual Depth Estimation](https://arxiv.org/abs/1812.07179)
2. [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)
3. [KITTI 3D Object Detection Benchmark](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
4. [Qualcomm Neural Processing SDK Documentation](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/introduction.html)
5. [CREStereo: Practical Stereo Matching via Cascaded Recurrent Network](https://arxiv.org/abs/2203.11483)

## 🤝 Contributing

When modifying this benchmark:
1. Maintain compatibility with SNPE SDK versions 2.x+
2. Test on actual QCS6490 hardware
3. Validate accuracy preservation after quantization
4. Document performance changes with hardware details
5. Follow established scientific methodology

---

**Note**: This benchmark requires the proprietary Qualcomm SNPE SDK. Performance results are highly dependent on thermal conditions, power supply stability, and SNPE SDK version. Always report these environmental factors with results.
