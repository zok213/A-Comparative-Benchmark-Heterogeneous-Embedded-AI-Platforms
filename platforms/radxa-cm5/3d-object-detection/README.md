# 3D Object Detection Benchmark - Radxa CM5 (RK3588S)

This directory contains the complete setup and execution scripts for benchmarking 3D Object Detection performance on the Radxa CM5 platform using the Pseudo-LiDAR + PointPillars pipeline with RKNN toolkit optimization.

## 📋 Overview

**Benchmark Type**: AI Accelerator Performance Evaluation  
**Target Platform**: Radxa CM5 with RK3588S + NPU + Mali GPU  
**Workload**: Pseudo-LiDAR + PointPillars 3D Object Detection Pipeline  
**Dataset**: KITTI 3D Object Detection  
**Key Metrics**: End-to-end Latency, Throughput (FPS), 3D mAP, Power Consumption  

## 🎯 Benchmark Objective

This benchmark evaluates the AI accelerator performance of the RK3588S platform using a two-stage 3D object detection pipeline optimized with RKNN toolkit. This demonstrates high-performance ARM-based embedded AI capabilities:

### Pipeline Architecture
1. **Stage 1**: Stereo depth estimation using CREStereo
2. **Intermediate**: Pseudo-LiDAR point cloud generation (CPU-based)
3. **Stage 2**: 3D object detection using PointPillars

### Why This Benchmark?
- **Real-world relevance**: Mimics autonomous vehicle perception pipelines
- **Heterogeneous compute**: Tests NPU, Mali GPU, and ARM CPU
- **Memory intensive**: Stresses memory bandwidth and cache hierarchy
- **ARM AI performance**: Evaluates dedicated NPU for AI workloads
- **RKNN optimization**: Demonstrates Rockchip's AI optimization toolkit

### RK3588S + NPU + Mali GPU Architecture
- **CPU**: 8-core ARM (4x Cortex-A76 @ 2.4 GHz + 4x Cortex-A55 @ 1.8 GHz)
- **GPU**: Mali-G610 MP4 GPU with OpenCL 2.2, Vulkan 1.2
- **AI Acceleration**: Dedicated 6 TOPS NPU
- **Memory**: LPDDR4/LPDDR4x/LPDDR5 support
- **TDP**: 5-15W (configurable)

## 🛠️ Prerequisites

### Hardware Requirements
- Radxa CM5 compute module with RK3588S
- 16GB LPDDR5 RAM (recommended) or 8GB minimum
- Active cooling solution (heatsink + fan) - **MANDATORY**
- 12V/2A DC power supply or USB-C PD (24W minimum)
- microSD card (128GB+) or eMMC storage (dataset is ~12GB)
- Yokogawa WT300E power analyzer (for accurate power measurement)

### Software Requirements
- Ubuntu 20.04 LTS (ARM64) or Debian 11
- RKNN Toolkit 1.5.x+ properly installed
- Mali GPU drivers for Linux
- Platform setup completed (run `../setup/install_all.sh` first)
- KITTI dataset downloaded (run `../../datasets/prepare_all_datasets.sh`)

## 🚀 Quick Start

### 1. Verify Prerequisites
```bash
# Check Mali GPU availability
ls /sys/class/misc/mali0/
cat /sys/class/misc/mali0/device/uevent

# Check NPU availability
ls /sys/kernel/debug/rknpu/
cat /sys/kernel/debug/rknpu/version

# Check if RKNN toolkit is properly installed
python3 -c "from rknn.api import RKNN; print('RKNN toolkit available')"

# Check if KITTI dataset is available
ls ~/benchmark_workspace/datasets/kitti/object/training/

# Check environment variables
source ~/benchmark_workspace/setup_env.sh
echo $MODELS_ROOT
echo $DATASETS_ROOT
```

### 2. Download Models and Datasets
```bash
# Download ONNX models (CREStereo + PointPillars)
cd radxa-cm5/3d-object-detection/
python3 download_models.py

# Download KITTI dataset using cookies (automatic with authentication)
python3 download_kitti_with_cookies.py

# Alternative: Manual KITTI download
# 1. Visit: https://www.cvlibs.net/datasets/kitti/
# 2. Login to your account
# 3. Download required files to ~/benchmark_workspace/datasets/kitti/
```

### 3. Run Benchmark
```bash
# After models and datasets are ready
chmod +x run_benchmark.sh
./run_benchmark.sh
```

### 4. View Results
```bash
# View NPU results
cat ~/benchmark_workspace/results/3d_detection/3d_detection_results_npu_*.json

# View Mali GPU results
cat ~/benchmark_workspace/results/3d_detection/3d_detection_results_gpu_*.json

# View ARM CPU results
cat ~/benchmark_workspace/results/3d_detection/3d_detection_results_cpu_*.json
```

## 📊 Understanding Results

### Output Files
```
~/benchmark_workspace/results/3d_detection/
├── 3d_detection_results_cpu_*.json    # CPU benchmark results
├── 3d_detection_results_gpu_*.json    # Intel UHD Graphics results
├── logs/                              # Execution logs
│   ├── cpu_benchmark.log
│   └── gpu_benchmark.log
├── calibration/                       # Calibration data
│   ├── image_2/                       # Left stereo images
│   ├── image_3/                       # Right stereo images
│   └── calib/                         # Camera calibration
└── models/
    └── openvino/                      # Converted OpenVINO models
        ├── crestereo_cpu/             # CREStereo optimized for CPU
        ├── crestereo_gpu/             # CREStereo optimized for GPU
        ├── pointpillars_cpu/          # PointPillars optimized for CPU
        └── pointpillars_gpu/          # PointPillars optimized for GPU
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

#### Device Comparison
- **CPU Runtime**: Uses Intel N100 CPU cores
- **GPU Runtime**: Uses Intel UHD Graphics with OpenVINO GPU plugin

### Expected Results (Reference)
Based on Intel N100 and UHD Graphics specifications:

| Component | CPU Runtime | GPU Runtime | Notes |
|-----------|-------------|-------------|-------|
| Stereo Estimation | 200-400 ms | 100-250 ms | GPU acceleration benefit |
| Point Cloud Conv. | 20-40 ms | 20-40 ms | CPU-only operation |
| 3D Detection | 300-600 ms | 150-400 ms | GPU acceleration benefit |
| **Total Pipeline** | **520-1040 ms** | **270-690 ms** | **End-to-end** |
| **Throughput** | **0.96-1.92 FPS** | **1.45-3.7 FPS** | **Sustained** |
| Power Consumption | 8-12 W | 10-15 W | Higher for GPU usage |

## ⚙️ Configuration Options

### Benchmark Parameters
Modify these parameters in the script:

```bash
NUM_ITERATIONS=1000         # Number of benchmark iterations
WARMUP_ITERATIONS=100       # Warmup iterations before measurement
```

### OpenVINO Optimization
The benchmark supports different runtime targets:
- **CPU**: Uses Intel N100 CPU cores with OpenVINO CPU plugin
- **GPU**: Uses Intel UHD Graphics with OpenVINO GPU plugin

### Model Optimization Features
- **Model Optimizer (MO)**: Converts ONNX to OpenVINO IR format
- **Post-Training Optimization Tool (POT)**: INT8 quantization
- **GPU-specific optimizations**: Optimized for Intel UHD Graphics architecture

## 🔧 Troubleshooting

### Common Issues

#### 1. OpenVINO Not Found
```bash
# Error: OpenVINO not found
# Solution: Install OpenVINO properly
pip install openvino openvino-dev

# Verify installation
python3 -c "import openvino; print(openvino.__version__)"
```

#### 2. Intel GPU Not Detected
```bash
# Check GPU availability
lspci | grep VGA
ls /sys/class/drm/card0/

# Install Intel GPU drivers if missing
sudo apt update
sudo apt install intel-gpu-tools
```

#### 3. Model Conversion Failures
```bash
# Check OpenVINO Model Optimizer
mo --help

# Manual model conversion
cd ~/benchmark_workspace/models/onnx/
mo --input_model crestereo.onnx --output_dir ../openvino/crestereo_fp32/
```

#### 4. KITTI Dataset Issues
```bash
# Verify dataset structure
tree ~/benchmark_workspace/datasets/kitti/object/ -L 3

# Re-download if corrupted
cd ../../datasets/
./prepare_all_datasets.sh
```

#### 5. Runtime Failures

**GPU Runtime Not Available**:
```bash
# Check OpenVINO GPU plugin
python3 -c "
from openvino.runtime import Core
core = Core()
print('Available devices:', core.available_devices)
"

# Fall back to CPU if GPU unavailable
# Edit run_benchmark.sh and comment out GPU benchmark
```

**Memory Issues**:
```bash
# Check available memory
free -h

# Reduce iterations if memory limited
export NUM_ITERATIONS=100
export WARMUP_ITERATIONS=10
```

**Performance Issues**:
- Ensure active cooling is working
- Check CPU/GPU frequencies
- Verify power supply adequacy (45W USB-C PD minimum)
- Monitor thermal throttling

### Debug Mode
Run with verbose output for troubleshooting:
```bash
bash -x run_benchmark.sh
```

### Manual Model Testing
Test individual models:
```bash
cd ~/benchmark_workspace/results/3d_detection/

# Test with OpenVINO benchmark app
benchmark_app -m ../models/openvino/crestereo_gpu/crestereo_fp32.xml -d GPU
```

## 🔬 Scientific Methodology

### Experimental Controls
- **Thermal Stability**: Active cooling prevents performance degradation
- **Model Consistency**: Identical ONNX models converted for each device
- **Dataset Consistency**: Same KITTI validation set for all tests
- **Statistical Rigor**: Multiple iterations with proper statistical analysis

### OpenVINO Optimization Process
- **Model Optimization**: ONNX to IR conversion with device-specific optimizations
- **Quantization**: INT8 post-training quantization using KITTI calibration data
- **Runtime Optimization**: Device-specific plugin optimizations (CPU vs GPU)

### Intel-Specific Considerations
- **Shared Memory Architecture**: CPU and GPU share LPDDR5 memory
- **Thermal Design Power**: 6W TDP may limit sustained performance
- **GPU Architecture**: 24 EUs optimized for compute workloads
- **Driver Dependencies**: Performance depends on Intel GPU driver version

## 📖 Advanced Usage

### Custom Device Testing
```bash
# Test specific OpenVINO devices
python3 benchmark_pipeline.py model_cpu model_gpu kitti_path AUTO  # Auto device selection
```

### Profiling and Analysis
```bash
# Enable OpenVINO profiling
export OV_CPU_PROFILE=1
export OV_GPU_PROFILE=1

# Run with profiling
./run_benchmark.sh

# Analyze performance bottlenecks
```

### Memory Usage Analysis
```bash
# Monitor memory usage during benchmark
sudo iotop -a &
sudo nethogs &
./run_benchmark.sh
```

### GPU Utilization Monitoring
```bash
# Monitor Intel GPU utilization
sudo intel_gpu_top &
./run_benchmark.sh
```

### Accuracy Evaluation
```bash
# Run accuracy evaluation (requires ground truth)
cd ~/benchmark_workspace/results/3d_detection/
python3 evaluate_3d_detection.py \
    --predictions detections/ \
    --ground_truth ../../datasets/kitti/object/training/label_2/
```

## 🔍 Performance Analysis

### CPU vs GPU Comparison
This benchmark enables analysis of:
- **Compute Efficiency**: CPU vs integrated GPU performance
- **Memory Bandwidth**: Shared memory architecture effects
- **Power Efficiency**: Performance per watt comparison
- **Thermal Behavior**: Sustained performance under thermal constraints

### x86 vs ARM Comparison
When compared with ARM platforms:
- **Architecture Differences**: x86 vs ARM instruction sets
- **Memory Systems**: Unified vs distributed memory architectures
- **AI Acceleration**: Integrated GPU vs dedicated NPU/DSP
- **Software Ecosystem**: OpenVINO vs TensorRT/SNPE

## 📚 References

1. [Pseudo-LiDAR from Visual Depth Estimation](https://arxiv.org/abs/1812.07179)
2. [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)
3. [KITTI 3D Object Detection Benchmark](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
4. [Intel OpenVINO Toolkit Documentation](https://docs.openvino.ai/)
5. [Intel N100 Processor Specifications](https://ark.intel.com/content/www/us/en/ark/products/231803/intel-processor-n100-6m-cache-up-to-3-40-ghz.html)
6. [Radxa X4 Technical Documentation](https://docs.radxa.com/en/x/x4)

## 🤝 Contributing

When modifying this benchmark:
1. Test on actual Radxa X4 hardware with Intel N100
2. Validate OpenVINO optimizations for both CPU and GPU
3. Ensure compatibility with OpenVINO 2023.x+
4. Document thermal conditions and cooling solutions
5. Follow established scientific methodology

---

**Note**: This benchmark requires Intel OpenVINO Toolkit and proper Intel GPU drivers. Performance results are highly dependent on thermal design, power delivery, and driver versions. Intel UHD Graphics performance may vary significantly based on memory bandwidth and thermal throttling. Always report cooling solution, ambient temperature, and power supply specifications with results.
