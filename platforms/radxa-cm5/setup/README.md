# Radxa CM5 Setup Guide

This directory contains the complete setup and installation scripts for running AI benchmarks on Radxa CM5 platforms with RK3588S processor.

## Overview

The setup process installs and configures all necessary dependencies for running ORB-SLAM3, 3D Object Detection, and Semantic Segmentation benchmarks on Radxa CM5 using RKNN toolkit for AI acceleration.

## Prerequisites

### Hardware Requirements
- **Radxa CM5** with RK3588S processor (8 cores: 4x A76 + 4x A55)
- **Mali-G610 MP4 GPU** (integrated GPU)
- **NPU**: 6 TOPS neural processing unit
- **Active Cooling**: Heatsink or fan (recommended for sustained performance)
- **Storage**: 64GB+ eMMC or high-speed microSD card (Class 10+)
- **Power Supply**: 12V/2A DC or USB-C PD (24W minimum)
- **Optional**: Power measurement equipment

### Software Requirements
- **Ubuntu 20.04 LTS** (ARM64 architecture) or Debian 11
- **Internet Connection**: Required for downloading RKNN toolkit and dependencies
- **Sudo Access**: Administrative privileges needed
- **Mali GPU Drivers**: Latest version for GPU acceleration

## Quick Setup

### Automated Installation
```bash
# Run the complete setup (recommended)
./install_all.sh

# Or run with specific options
./install_all.sh --skip-datasets --verbose
```

### Manual Step-by-Step
```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install RKNN toolkit
./install_all.sh --rknn-only

# 3. Install benchmark-specific dependencies
./install_all.sh --benchmarks-only
```

## Installation Components

### System Dependencies
- **Build Tools**: GCC, CMake, Git, Python 3.8+
- **Mali Graphics**: Mali GPU drivers, OpenCL support
- **System Libraries**: OpenCV, NumPy, SciPy, Matplotlib
- **Development Headers**: Linux kernel headers, build-essential
- **Networking**: wget, curl for downloading dependencies
- **RK3588S Specific**: Device tree compiler, Rockchip MPP, Mali firmware

### RKNN Toolkit
- **Runtime**: RKNN inference engine
- **Model Converter**: Tool for converting ONNX/TensorFlow/PyTorch to RKNN format
- **Quantization**: INT8 quantization for NPU acceleration
- **Multi-target Support**: NPU, Mali GPU, and ARM CPU backends
- **Python API**: Python bindings for RKNN toolkit

### Benchmark-Specific Dependencies

#### ORB-SLAM3
- **Eigen3**: Linear algebra library (ARM optimized)
- **Pangolin**: 3D visualization library
- **DBoW2**: Bag-of-words library
- **g2o**: Graph optimization library (ARM optimized build)

#### 3D Object Detection
- **RKNN Models**: CREStereo and PointPillars optimized for NPU/GPU/CPU
- **NumPy**: Numerical computing with ARM optimizations
- **SciPy**: Scientific computing library

#### Semantic Segmentation
- **Pillow**: Python imaging library
- **scikit-image**: Image processing toolkit
- **OpenCV**: Computer vision with ARM optimizations

## RKNN Toolkit Installation

### Download and Setup
```bash
# Create RKNN installation directory
mkdir -p ~/rknn_toolkit
cd ~/rknn_toolkit

# Download RKNN toolkit (latest version for ARM64)
wget https://github.com/rockchip-linux/rknn-toolkit2/releases/download/v1.5.2/rknn_toolkit2-1.5.2-cp38-cp38-linux_aarch64.whl

# Install RKNN toolkit
pip3 install rknn_toolkit2-1.5.2-cp38-cp38-linux_aarch64.whl

# Install additional dependencies
pip3 install onnx onnx-simplifier tensorflow torch torchvision numpy opencv-python
```

### NPU and GPU Driver Setup
```bash
# Check NPU availability
ls /sys/kernel/debug/rknpu/
cat /sys/kernel/debug/rknpu/version

# Check Mali GPU availability
ls /sys/class/misc/mali0/
cat /sys/class/misc/mali0/device/uevent

# Install Mali GPU support (if needed)
sudo apt install -y mali-g610-firmware opencl-headers ocl-icd-opencl-dev clinfo
```

## Configuration Options

### Environment Variables
```bash
# OpenVINO configuration
export INTEL_OPENVINO_DIR="/opt/intel/openvino"
export LD_LIBRARY_PATH="$INTEL_OPENVINO_DIR/runtime/lib/intel64:$LD_LIBRARY_PATH"
export PYTHONPATH="$INTEL_OPENVINO_DIR/python/python3.8:$PYTHONPATH"

# Intel optimizations
export OMP_NUM_THREADS=4      # CPU threading
export KMP_AFFINITY=granularity=fine,compact,1,0
export MKL_NUM_THREADS=4      # Intel MKL threading

# Performance settings
export RADXA_X4_CLOCKS=1      # Enable performance clocks
export GPU_ACCELERATION=1     # Enable GPU acceleration
export CPU_OPTIMIZATION=1     # Enable CPU optimizations
```

### Installation Flags
```bash
./install_all.sh [OPTIONS]

Options:
  --help                Show this help message
  --verbose            Enable verbose output
  --skip-system        Skip system package installation
  --skip-openvino      Skip OpenVINO installation
  --skip-gpu-drivers   Skip Intel GPU driver installation
  --skip-python        Skip Python environment setup
  --skip-benchmarks    Skip benchmark-specific dependencies
  --skip-datasets      Skip dataset preparation
  --rebuild-opencv     Force rebuild OpenCV with Intel optimizations
  --rebuild-orbslam3   Force rebuild ORB-SLAM3
  --openvino-only      Install only OpenVINO toolkit
  --benchmarks-only    Install only benchmark dependencies
  --clean              Clean previous installations
```

## Performance Optimization

### System Configuration
The setup script automatically configures:

#### CPU Performance
```bash
# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Set maximum CPU frequencies
sudo cpufreq-set -c 0-3 -g performance

# Enable Intel Turbo Boost (if available)
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

#### GPU Configuration
```bash
# Set GPU to maximum performance
echo 1 | sudo tee /sys/class/drm/card0/gt_max_freq_mhz

# Check GPU frequency
cat /sys/class/drm/card0/gt_cur_freq_mhz
cat /sys/class/drm/card0/gt_max_freq_mhz
```

#### Memory Optimization
```bash
# Configure memory governor
echo performance | sudo tee /sys/devices/system/cpu/cpufreq/scaling_governor

# Increase swap space if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Intel Optimizations
```bash
# Enable Intel MKL optimizations
export MKL_ENABLE_INSTRUCTIONS=AVX2
export MKL_THREADING_LAYER=GNU

# OpenMP optimizations
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

## Verification

### System Verification
```bash
# Check OpenVINO installation
python3 -c "from openvino.runtime import Core; print('OpenVINO available')"

# Check available devices
python3 -c "from openvino.runtime import Core; print(Core().available_devices)"

# Check Intel GPU
intel_gpu_top  # If available
clinfo | grep "Device Name"

# Check OpenCV installation
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

### Benchmark Verification
```bash
# Test ORB-SLAM3 build
cd ../orb-slam3
python3 scripts/test_orbslam3.py

# Test OpenVINO model conversion
cd ../3d-object-detection
python3 scripts/test_openvino_conversion.py

# Test segmentation models
cd ../semantic-segmentation
python3 scripts/test_segmentation.py
```

## Troubleshooting

### Common Installation Issues

#### OpenVINO Installation Issues
```bash
# Check OpenVINO environment
echo $INTEL_OPENVINO_DIR
ls $INTEL_OPENVINO_DIR/runtime/lib/intel64/

# Verify Python bindings
python3 -c "import openvino; print(openvino.__version__)"
```

#### GPU Driver Issues
```bash
# Check Intel GPU driver
lsmod | grep i915

# Check GPU device files
ls -la /dev/dri/

# Test GPU functionality
intel_gpu_top
```

#### Performance Issues
```bash
# Check CPU governor
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Check CPU frequencies
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq

# Check GPU frequency
cat /sys/class/drm/card0/gt_cur_freq_mhz
```

#### Memory Issues
```bash
# Check memory usage
free -h
cat /proc/meminfo

# Check for memory leaks
valgrind --leak-check=full python3 test_script.py
```

### Performance Debugging
```bash
# Monitor CPU usage
htop

# Monitor GPU usage (if available)
intel_gpu_top

# Profile with perf
perf record -g python3 benchmark_script.py
perf report
```

## Advanced Configuration

### Custom OpenVINO Build
```bash
# Build OpenVINO from source (advanced users)
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino
git submodule update --init --recursive
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

### Development Environment
```bash
# Install debugging tools
sudo apt install gdb valgrind strace

# Install profiling tools
sudo apt install perf-tools-unstable

# Enable core dumps
ulimit -c unlimited
```

### Intel Compiler (Optional)
```bash
# Install Intel oneAPI (optional for maximum performance)
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-basekit
```

## Maintenance

### Regular Updates
```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Update Python packages
pip3 install --upgrade numpy scipy matplotlib pillow

# Check for OpenVINO updates
# Visit: https://github.com/openvinotoolkit/openvino/releases
```

### Clean Installation
```bash
# Remove all benchmark dependencies
./install_all.sh --clean

# Full system cleanup
sudo apt autoremove && sudo apt autoclean
```

## Performance Monitoring

### System Monitoring
```bash
# Monitor CPU/GPU usage
htop
intel_gpu_top  # If available

# Monitor power consumption (if supported)
cat /sys/class/power_supply/*/power_now

# Monitor temperatures
sensors  # If lm-sensors installed
cat /sys/class/thermal/thermal_zone*/temp
```

### OpenVINO Profiling
```bash
# Use OpenVINO benchmark app
benchmark_app -m model.xml -d CPU
benchmark_app -m model.xml -d GPU

# Enable detailed profiling
benchmark_app -m model.xml -d CPU -pc
```

## Support

For installation issues:
1. Check the troubleshooting section above
2. Review Intel OpenVINO documentation: https://docs.openvino.ai/
3. Consult Intel Developer Forums: https://community.intel.com/
4. Check system logs: `dmesg` and `/var/log/syslog`

## References

- [Intel OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- [Intel N100 Processor](https://ark.intel.com/content/www/us/en/ark/products/231803/intel-processor-n100-6m-cache-up-to-3-40-ghz.html)
- [Radxa X4 Documentation](https://docs.radxa.com/en/x/x4)
- [Intel Graphics Driver Installation](https://dgpu-docs.intel.com/driver/installation.html)
