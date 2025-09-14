# NVIDIA Jetson Setup Guide

This directory contains the complete setup and installation scripts for running AI benchmarks on NVIDIA Jetson platforms.

## Overview

The setup process installs and configures all necessary dependencies for running ORB-SLAM3, 3D Object Detection, and Semantic Segmentation benchmarks on NVIDIA Jetson Orin NX platforms.

## Prerequisites

### Hardware Requirements
- **NVIDIA Jetson Orin NX** (8GB or 16GB variant)
- **Active Cooling**: Fan or heatsink (critical for sustained performance)
- **Storage**: MicroSD card (128GB+, Class 10 or better) or NVMe SSD
- **Power Supply**: Official NVIDIA power adapter or compatible 19V supply
- **Optional**: Yokogawa WT300E power meter for power measurement

### Software Requirements
- **JetPack 5.1.1** (Ubuntu 20.04 LTS base)
- **Internet Connection**: Required for downloading dependencies
- **Sudo Access**: Administrative privileges needed

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

# 2. Install CUDA development tools
sudo apt install cuda-toolkit-11-4 cuda-samples-11-4

# 3. Install deep learning frameworks
./install_all.sh --frameworks-only

# 4. Install benchmark-specific dependencies
./install_all.sh --benchmarks-only
```

## Installation Components

### System Dependencies
- **Build Tools**: GCC, CMake, Git, Python 3.8+
- **CUDA Toolkit**: CUDA 11.4+ with development headers
- **cuDNN**: NVIDIA Deep Neural Network library
- **TensorRT**: NVIDIA inference optimization library
- **OpenCV**: Computer vision library with CUDA support

### Python Environment
- **PyTorch**: Deep learning framework with CUDA support
- **ONNX**: Open Neural Network Exchange format support
- **NumPy, SciPy**: Scientific computing libraries
- **Matplotlib**: Plotting and visualization
- **Pandas**: Data analysis and manipulation

### Benchmark-Specific Dependencies

#### ORB-SLAM3
- **Eigen3**: Linear algebra library
- **Pangolin**: 3D visualization library
- **DBoW2**: Bag-of-words library
- **g2o**: Graph optimization library

#### 3D Object Detection
- **PCL**: Point Cloud Library
- **Open3D**: 3D data processing
- **TensorRT Python API**: Python bindings for TensorRT

#### Semantic Segmentation
- **Pillow**: Python imaging library
- **scikit-image**: Image processing toolkit
- **torchvision**: PyTorch computer vision utilities

## Configuration Options

### Environment Variables
The installation script supports several configuration options:

```bash
# Installation paths
export JETSON_INSTALL_PREFIX="/opt/jetson-benchmarks"
export CUDA_HOME="/usr/local/cuda"
export TENSORRT_ROOT="/usr/src/tensorrt"

# Build options
export ENABLE_CUDA=1
export ENABLE_TENSORRT=1
export OPENCV_CUDA_SUPPORT=1
export PYTORCH_CUDA_ARCH_LIST="8.7"  # Orin NX architecture

# Performance settings
export JETSON_CLOCKS=1  # Enable maximum clocks
export POWER_MODE="MAXN"  # Maximum performance mode
```

### Installation Flags
```bash
./install_all.sh [OPTIONS]

Options:
  --help                Show this help message
  --verbose            Enable verbose output
  --skip-system        Skip system package installation
  --skip-cuda          Skip CUDA toolkit installation
  --skip-tensorrt      Skip TensorRT installation
  --skip-python        Skip Python environment setup
  --skip-benchmarks    Skip benchmark-specific dependencies
  --skip-datasets      Skip dataset preparation
  --rebuild-opencv     Force rebuild OpenCV with CUDA
  --rebuild-orbslam3   Force rebuild ORB-SLAM3
  --clean              Clean previous installations
```

## Verification

### System Verification
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Check TensorRT
python3 -c "import tensorrt; print(tensorrt.__version__)"

# Check PyTorch CUDA support
python3 -c "import torch; print(torch.cuda.is_available())"

# Check OpenCV CUDA support
python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
```

### Benchmark Verification
```bash
# Test ORB-SLAM3 build
cd ../orb-slam3
python3 scripts/test_orbslam3.py

# Test 3D detection models
cd ../3d-object-detection
python3 scripts/test_models.py

# Test segmentation models
cd ../semantic-segmentation
python3 scripts/test_segmentation.py
```

## Performance Optimization

### System Configuration
The setup script automatically configures:

#### Power Management
```bash
# Set maximum performance mode
sudo nvpmodel -m 0  # MAXN mode

# Enable jetson_clocks
sudo jetson_clocks

# Disable CPU idle states (for consistent performance)
echo 1 | sudo tee /sys/devices/system/cpu/cpu*/cpuidle/state*/disable
```

#### Memory Configuration
```bash
# Increase swap space (if needed)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Configure memory frequency
echo performance | sudo tee /sys/class/devfreq/17000000.gv11b/governor
```

#### Thermal Management
```bash
# Configure fan control (if available)
echo 255 | sudo tee /sys/devices/pwm-fan/target_pwm

# Monitor thermal zones
cat /sys/class/thermal/thermal_zone*/temp
```

## Troubleshooting

### Common Installation Issues

#### CUDA Installation Problems
```bash
# Check CUDA installation
ls -la /usr/local/cuda*

# Reinstall CUDA toolkit
sudo apt remove --purge cuda*
sudo apt install cuda-toolkit-11-4
```

#### TensorRT Issues
```bash
# Check TensorRT installation
dpkg -l | grep tensorrt

# Reinstall TensorRT
sudo apt remove --purge libnvinfer*
sudo apt install tensorrt
```

#### OpenCV Build Failures
```bash
# Clean and rebuild OpenCV
./install_all.sh --rebuild-opencv --verbose

# Check OpenCV build log
tail -f /tmp/opencv_build.log
```

#### Python Environment Issues
```bash
# Reset Python environment
pip3 uninstall torch torchvision -y
./install_all.sh --skip-system --skip-cuda

# Check Python paths
python3 -c "import sys; print(sys.path)"
```

### Performance Issues

#### Low GPU Utilization
```bash
# Check power mode
sudo nvpmodel -q

# Enable maximum clocks
sudo jetson_clocks --show
sudo jetson_clocks
```

#### Memory Issues
```bash
# Check memory usage
free -h
nvidia-smi

# Increase swap if needed
sudo swapoff /swapfile
sudo fallocate -l 16G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Thermal Throttling
```bash
# Monitor temperatures
sudo tegrastats

# Check thermal zones
cat /sys/class/thermal/thermal_zone*/temp

# Verify cooling solution
```

## Advanced Configuration

### Custom CUDA Installation
```bash
# Install specific CUDA version
export CUDA_VERSION="11.4"
./install_all.sh --cuda-version=$CUDA_VERSION
```

### Development Environment
```bash
# Install additional development tools
sudo apt install nsight-systems nsight-compute
sudo apt install valgrind gdb

# Enable core dumps
ulimit -c unlimited
```

### Cross-Compilation Setup
```bash
# Setup for cross-compilation (if needed)
export CROSS_COMPILE=aarch64-linux-gnu-
export CC=aarch64-linux-gnu-gcc
export CXX=aarch64-linux-gnu-g++
```

## Maintenance

### Regular Updates
```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Update Python packages
pip3 install --upgrade torch torchvision onnx

# Update TensorRT (when available)
sudo apt update && sudo apt upgrade tensorrt
```

### Clean Installation
```bash
# Remove all benchmark dependencies
./install_all.sh --clean

# Full system cleanup
sudo apt autoremove && sudo apt autoclean
```

## Support

For installation issues:
1. Check the troubleshooting section above
2. Review JetPack documentation: https://docs.nvidia.com/jetson/
3. Consult NVIDIA Developer Forums: https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/
4. Check system logs: `dmesg` and `/var/log/syslog`

## References

- [JetPack SDK](https://developer.nvidia.com/embedded/jetpack)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Jetson Performance Tuning](https://docs.nvidia.com/jetson/archives/r35.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance.html)
