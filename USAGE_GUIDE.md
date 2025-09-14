# Complete Usage Guide - Embedded AI Benchmark Suite

This comprehensive guide walks you through the entire process of setting up and running the embedded AI benchmark suite on all supported platforms.

## 🎯 Quick Start

### Prerequisites Check
Before starting, ensure you have:
- ✅ Target hardware platform(s)
- ✅ Active cooling solutions installed
- ✅ Power measurement equipment (optional but recommended)
- ✅ Host PC with Ubuntu 20.04 LTS
- ✅ Network connectivity for downloads

### One-Command Setup and Run
```bash
# Clone the repository
git clone <repository-url>
cd embedded-ai-benchmark-suite

# Make the master script executable
chmod +x run_all_benchmarks.sh

# Run complete benchmark suite with setup
./run_all_benchmarks.sh --setup --datasets --analysis --power
```

## 📋 Step-by-Step Guide

### Step 1: Platform Setup

#### For NVIDIA Jetson Orin NX:
```bash
cd nvidia-jetson/setup/
chmod +x install_all.sh
./install_all.sh

# Reboot after setup
sudo reboot
```

#### For Qualcomm QCS6490:
```bash
cd qualcomm-qcs6490/setup/
chmod +x install_all.sh
./install_all.sh

# Note: SNPE SDK must be manually downloaded first
# See setup script output for instructions
```

#### For Radxa X4 (Intel N100):
```bash
cd radxa-x4/setup/
chmod +x install_all.sh
./install_all.sh

# Reboot after setup
sudo reboot
```

### Step 2: Dataset Preparation
```bash
cd datasets/
chmod +x prepare_all_datasets.sh
./prepare_all_datasets.sh

# Follow instructions for manual downloads:
# - KITTI dataset (requires registration)
# - Cityscapes dataset (requires registration)
# - EuRoC MAV dataset (automatic download)
```

### Step 3: Run Individual Benchmarks

#### ORB-SLAM3 (CPU Benchmark):
```bash
cd nvidia-jetson/orb-slam3/    # or qualcomm-qcs6490/ or radxa-x4/
chmod +x run_benchmark.sh
./run_benchmark.sh
```

#### 3D Object Detection (AI Accelerator Benchmark):
```bash
cd nvidia-jetson/3d-object-detection/    # or qualcomm-qcs6490/ or radxa-x4/
chmod +x run_benchmark.sh
./run_benchmark.sh
```

#### Semantic Segmentation (AI Accelerator Benchmark):
```bash
cd nvidia-jetson/semantic-segmentation/    # or qualcomm-qcs6490/ or radxa-x4/
chmod +x run_benchmark.sh
./run_benchmark.sh
```

### Step 4: Results Analysis
```bash
cd analysis/scripts/
python3 analyze_all_results.py --results-root ../.. --output-dir ../results
```

## 🔧 Advanced Usage

### Selective Platform/Benchmark Execution
```bash
# Run only specific platforms
./run_all_benchmarks.sh --platforms nvidia-jetson,radxa-x4

# Run only specific benchmarks
./run_all_benchmarks.sh --benchmarks orb-slam3,semantic-segmentation

# Run with power logging
./run_all_benchmarks.sh --power --benchmarks semantic-segmentation
```

### Manual Model Optimization

#### NVIDIA Jetson (TensorRT):
```bash
# Convert ONNX to TensorRT engine
/usr/src/tensorrt/bin/trtexec \
    --onnx=model.onnx \
    --saveEngine=model_gpu.engine \
    --int8 \
    --calib=calibration.cache \
    --workspace=4096

# For DLA acceleration
/usr/src/tensorrt/bin/trtexec \
    --onnx=model.onnx \
    --saveEngine=model_dla.engine \
    --int8 \
    --calib=calibration.cache \
    --useDLACore=0 \
    --allowGPUFallback
```

#### Qualcomm QCS6490 (SNPE):
```bash
# Convert ONNX to DLC
snpe-onnx-to-dlc --input_network model.onnx --output_path model_fp32.dlc

# Quantize for Hexagon NPU
snpe-dlc-quantize \
    --input_dlc model_fp32.dlc \
    --input_list calibration_list.txt \
    --output_dlc model_quantized.dlc \
    --enable_htp
```

#### Radxa X4 (OpenVINO):
```bash
# Convert ONNX to OpenVINO IR
mo --input_model model.onnx --output_dir fp16_model/ --data_type FP16

# Quantize to INT8 using NNCF
python3 quantize_model.py fp16_model/model.xml calibration_dir/ int8_model/
```

## 📊 Understanding Results

### Result File Structure
```
platform-name/results/
├── orb_slam3/
│   ├── detailed_analysis.txt          # Performance metrics
│   ├── performance_analysis.png       # Visualization
│   └── logs/                          # Individual run logs
├── 3d_detection/
│   ├── 3d_detection_results_gpu_*.json    # GPU results
│   ├── 3d_detection_results_dla_*.json    # DLA results (NVIDIA only)
│   └── logs/                              # Execution logs
└── semantic-segmentation/
    ├── segmentation_results_gpu_*.json     # GPU results
    ├── segmentation_results_dsp_*.json     # NPU results (Qualcomm only)
    └── logs/                               # Execution logs
```

### Key Metrics Explained

#### Performance Metrics:
- **Throughput (FPS)**: Frames processed per second
- **P99 Latency**: 99th percentile latency (worst-case performance)
- **Mean Latency**: Average processing time per frame

#### Efficiency Metrics:
- **Power (W)**: Average power consumption during benchmark
- **Efficiency (FPS/W)**: Performance per watt consumed

#### Accuracy Metrics:
- **mIoU**: Mean Intersection over Union (semantic segmentation)
- **3D mAP**: 3D mean Average Precision (object detection)
- **ATE/RTE**: Absolute/Relative Trajectory Error (SLAM)

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. Setup Script Failures
```bash
# Check system requirements
lsb_release -a    # Should be Ubuntu 20.04
uname -m          # Check architecture

# Clean previous installations
sudo apt autoremove
sudo apt autoclean

# Re-run setup with verbose output
bash -x install_all.sh
```

#### 2. Model Download Failures
```bash
# Manual model download
wget https://huggingface.co/qualcomm/DDRNet23-Slim/resolve/main/DDRNet23-Slim.onnx
mv DDRNet23-Slim.onnx models/onnx/ddrnet23-slim.onnx
```

#### 3. Dataset Issues
```bash
# Verify dataset structure
tree datasets/ -L 3

# Re-extract datasets
cd datasets/kitti/
unzip -o data_object_image_2.zip
```

#### 4. Benchmark Execution Failures
```bash
# Check environment
source ~/benchmark_workspace/setup_env.sh
echo $BENCHMARK_ROOT
echo $DATASETS_ROOT

# Check system resources
free -h           # Memory
df -h             # Disk space
nvidia-smi        # GPU (NVIDIA only)
```

#### 5. Power Measurement Issues
```bash
# Check power analyzer connection
ping 192.168.1.100    # Default IP

# Verify measurement setup
python3 -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = s.connect_ex(('192.168.1.100', 502))
print('Connected' if result == 0 else 'Connection failed')
s.close()
"
```

### Platform-Specific Issues

#### NVIDIA Jetson:
```bash
# Check JetPack version
jetson_release

# Verify TensorRT installation
/usr/src/tensorrt/bin/trtexec --help

# Check thermal throttling
sudo tegrastats

# Reset clocks
sudo nvpmodel -m 0
sudo jetson_clocks
```

#### Qualcomm QCS6490:
```bash
# Verify SNPE installation
echo $SNPE_ROOT
ls $SNPE_ROOT/bin/aarch64-linux-gcc7.5/

# Check DSP availability
cat /sys/kernel/debug/msm_subsys/adsp

# Monitor system
cat /proc/cpuinfo | grep processor
```

#### Radxa X4:
```bash
# Check OpenVINO installation
python3 -c "import openvino as ov; print(ov.__version__)"

# Verify Intel GPU
clinfo -l

# Check Intel GPU driver
ls /dev/dri/

# Monitor CPU frequency
watch -n 1 "cat /proc/cpuinfo | grep MHz"
```

## 🏃‍♂️ Performance Optimization Tips

### System Optimization
```bash
# Set performance governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable swap (if sufficient RAM)
sudo swapoff -a

# Increase file descriptor limits
ulimit -n 65536

# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable cups
```

### Thermal Management
```bash
# Monitor temperatures
watch -n 1 "sensors | grep temp"

# For NVIDIA Jetson
watch -n 1 "cat /sys/devices/virtual/thermal/thermal_zone*/temp"

# Ensure active cooling
# - Check fan operation
# - Verify heatsink contact
# - Monitor for thermal throttling
```

### Memory Optimization
```bash
# Clear caches before benchmarks
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches

# Monitor memory usage
watch -n 1 "free -h"

# For large models, ensure sufficient swap
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## 📈 Results Interpretation

### Comparative Analysis
The benchmark suite generates several types of comparisons:

1. **Cross-Platform Performance**: Compare the same workload across different hardware
2. **Cross-Accelerator Performance**: Compare different accelerators on the same platform
3. **Performance vs. Power Trade-offs**: Analyze efficiency characteristics
4. **Accuracy vs. Performance**: Understand quantization impact

### Expected Performance Ranges

#### ORB-SLAM3 (CPU-bound):
- **NVIDIA Jetson Orin NX**: 15-25 FPS
- **Qualcomm QCS6490**: 12-20 FPS
- **Radxa X4 (Intel N100)**: 18-28 FPS

#### Semantic Segmentation (GPU/NPU):
- **NVIDIA GPU**: 10-20 FPS
- **NVIDIA DLA**: 5-12 FPS
- **Qualcomm Hexagon**: 8-15 FPS
- **Intel UHD Graphics**: 6-12 FPS

#### 3D Object Detection (Pipeline):
- **End-to-end latency**: 200-500ms
- **Accuracy (3D mAP)**: 15-25% (depends on stereo quality)

### Validation Checklist
- [ ] Multiple runs show consistent results (CV < 10%)
- [ ] No thermal throttling during benchmarks
- [ ] Power measurements are stable
- [ ] Accuracy metrics are within expected ranges
- [ ] System resources (memory, storage) are adequate

## 🚀 Next Steps

### Extending the Benchmark Suite
1. **Add New Platforms**: Follow the existing platform structure
2. **Add New Benchmarks**: Implement similar benchmark scripts
3. **Add New Models**: Use the same optimization pipelines
4. **Custom Analysis**: Extend the analysis scripts

### Research Applications
- **Hardware Selection**: Use results to choose optimal platforms
- **Algorithm Optimization**: Identify bottlenecks and optimization opportunities
- **Power Budget Planning**: Design systems within power constraints
- **Performance Prediction**: Model performance for new workloads

For additional support and updates, please refer to the project documentation and community forums.
