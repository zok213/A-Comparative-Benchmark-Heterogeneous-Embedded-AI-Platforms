# Qualcomm QCS6490 Setup Guide

This directory contains the complete setup and installation scripts for running AI benchmarks on Qualcomm QCS6490 platforms.

## Overview

The setup process installs and configures all necessary dependencies for running ORB-SLAM3, 3D Object Detection, and Semantic Segmentation benchmarks on Qualcomm QCS6490 (or Robotics RB5) platforms using the Qualcomm Neural Processing SDK (SNPE).

## Prerequisites

### Hardware Requirements
- **Qualcomm QCS6490** or **Robotics RB5 Development Kit**
- **Active Cooling**: Heatsink or fan (recommended for sustained performance)
- **Storage**: 64GB+ eUFS storage or high-speed microSD card (Class 10+)
- **Power Supply**: 12V/3A power adapter (official recommended)
- **Optional**: Yokogawa WT300E power meter for power measurement

### Software Requirements
- **Ubuntu 20.04 LTS** (or compatible ARM64 Linux distribution)
- **Internet Connection**: Required for downloading SNPE SDK and dependencies
- **Sudo Access**: Administrative privileges needed
- **Qualcomm Developer Account**: For SNPE SDK download

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

# 2. Install SNPE SDK
./install_all.sh --snpe-only

# 3. Install benchmark-specific dependencies
./install_all.sh --benchmarks-only
```

## Installation Components

### System Dependencies
- **Build Tools**: GCC, CMake, Git, Python 3.8+
- **System Libraries**: OpenCV, NumPy, SciPy, Matplotlib
- **Development Headers**: Linux kernel headers, build-essential
- **Networking**: wget, curl for downloading dependencies

### Qualcomm SNPE SDK
- **SNPE Runtime**: Neural processing runtime for CPU/GPU/DSP
- **Model Converter**: Tools for converting ONNX/TensorFlow models to DLC
- **Platform Validator**: Runtime capability verification
- **Profiling Tools**: Performance analysis utilities
- **Python API**: Python bindings for SNPE runtime

### Benchmark-Specific Dependencies

#### ORB-SLAM3
- **Eigen3**: Linear algebra library (ARM64 optimized)
- **Pangolin**: 3D visualization library
- **DBoW2**: Bag-of-words library
- **g2o**: Graph optimization library (ARM64 build)

#### 3D Object Detection
- **Open3D**: 3D data processing (ARM64 build)
- **NumPy**: Numerical computing with ARM optimizations
- **SciPy**: Scientific computing library

#### Semantic Segmentation
- **Pillow**: Python imaging library
- **scikit-image**: Image processing toolkit
- **OpenCV**: Computer vision with ARM NEON optimizations

## SNPE SDK Installation

### Download and Setup
```bash
# Download SNPE SDK (requires Qualcomm developer account)
# Visit: https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk

# Extract and install
export SNPE_ROOT="/opt/qcom/aistack/snpe"
sudo mkdir -p $SNPE_ROOT
sudo tar -xzf snpe-*.tgz -C $SNPE_ROOT --strip-components=1

# Set up environment
echo "export SNPE_ROOT=$SNPE_ROOT" >> ~/.bashrc
echo "export PATH=\$PATH:\$SNPE_ROOT/bin/aarch64-ubuntu-gcc7.5" >> ~/.bashrc
source ~/.bashrc
```

### Verification
```bash
# Check SNPE installation
snpe-platform-validator

# Test runtimes
snpe-platform-validator --runtime cpu
snpe-platform-validator --runtime gpu
snpe-platform-validator --runtime dsp
```

## Configuration Options

### Environment Variables
```bash
# SNPE configuration
export SNPE_ROOT="/opt/qcom/aistack/snpe"
export SNPE_TARGET_ARCH="aarch64-ubuntu-gcc7.5"
export LD_LIBRARY_PATH="$SNPE_ROOT/lib/aarch64-ubuntu-gcc7.5:$LD_LIBRARY_PATH"

# Performance settings
export QCS6490_CLOCKS=1  # Enable performance clocks
export DSP_RUNTIME=1     # Enable DSP runtime
export GPU_RUNTIME=1     # Enable GPU runtime

# Build options
export ENABLE_NEON=1     # Enable ARM NEON optimizations
export CMAKE_BUILD_TYPE="Release"
```

### Installation Flags
```bash
./install_all.sh [OPTIONS]

Options:
  --help                Show this help message
  --verbose            Enable verbose output
  --skip-system        Skip system package installation
  --skip-snpe          Skip SNPE SDK setup
  --skip-python        Skip Python environment setup
  --skip-benchmarks    Skip benchmark-specific dependencies
  --skip-datasets      Skip dataset preparation
  --rebuild-opencv     Force rebuild OpenCV with optimizations
  --rebuild-orbslam3   Force rebuild ORB-SLAM3
  --snpe-only          Install only SNPE SDK
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
sudo cpufreq-set -c 0-7 -g performance

# Disable CPU idle states for consistent performance
echo 1 | sudo tee /sys/devices/system/cpu/cpu*/cpuidle/state*/disable
```

#### DSP Configuration
```bash
# Enable DSP runtime
echo 1 | sudo tee /sys/kernel/debug/msm_subsys/slpi

# Check DSP status
cat /sys/kernel/debug/msm_subsys/slpi
```

#### GPU Configuration
```bash
# Set GPU governor to performance
echo performance | sudo tee /sys/class/kgsl/kgsl-3d0/devfreq/governor

# Set maximum GPU frequency
cat /sys/class/kgsl/kgsl-3d0/devfreq/max_freq | sudo tee /sys/class/kgsl/kgsl-3d0/devfreq/min_freq
```

#### Memory Optimization
```bash
# Configure memory frequency
echo performance | sudo tee /sys/class/devfreq/soc:qcom,memlat-cpu*/governor

# Increase swap space if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Verification

### System Verification
```bash
# Check SNPE installation
snpe-platform-validator

# Check runtime availability
snpe-platform-validator --runtime dsp
snpe-platform-validator --runtime gpu
snpe-platform-validator --runtime cpu

# Check Python SNPE bindings
python3 -c "import snpe; print('SNPE Python API available')"

# Check OpenCV installation
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

### Benchmark Verification
```bash
# Test ORB-SLAM3 build
cd ../orb-slam3
python3 scripts/test_orbslam3.py

# Test SNPE model conversion
cd ../3d-object-detection
python3 scripts/test_snpe_conversion.py

# Test segmentation models
cd ../semantic-segmentation
python3 scripts/test_segmentation.py
```

## Troubleshooting

### Common Installation Issues

#### SNPE SDK Issues
```bash
# Check SNPE environment
echo $SNPE_ROOT
ls $SNPE_ROOT/lib/aarch64-ubuntu-gcc7.5/

# Verify library paths
ldd $SNPE_ROOT/bin/aarch64-ubuntu-gcc7.5/snpe-net-run
```

#### DSP Runtime Issues
```bash
# Check DSP subsystem
cat /sys/kernel/debug/msm_subsys/slpi

# Check DSP firmware
ls /lib/firmware/

# Restart DSP subsystem (if needed)
echo restart | sudo tee /sys/kernel/debug/msm_subsys/slpi
```

#### GPU Runtime Issues
```bash
# Check Adreno GPU status
cat /sys/class/kgsl/kgsl-3d0/gpubusy

# Check GPU driver
lsmod | grep kgsl

# Verify GPU frequencies
cat /sys/class/kgsl/kgsl-3d0/devfreq/available_frequencies
```

#### OpenCV Build Issues
```bash
# Clean and rebuild OpenCV
./install_all.sh --rebuild-opencv --verbose

# Check OpenCV build configuration
python3 -c "import cv2; print(cv2.getBuildInformation())"
```

### Performance Issues

#### Low DSP Performance
```bash
# Check DSP clocks
cat /sys/kernel/debug/clk/clk_summary | grep dsp

# Monitor DSP utilization
cat /sys/kernel/debug/msm_subsys/slpi
```

#### Memory Issues
```bash
# Check memory usage
free -h
cat /proc/meminfo

# Monitor memory pressure
cat /proc/pressure/memory
```

#### Thermal Throttling
```bash
# Monitor temperatures
cat /sys/class/thermal/thermal_zone*/temp

# Check thermal zones
ls /sys/class/thermal/thermal_zone*/type
```

## Advanced Configuration

### Custom SNPE Build
```bash
# Build SNPE from source (if available)
export SNPE_SOURCE_DIR="/path/to/snpe/source"
cd $SNPE_SOURCE_DIR
make aarch64-ubuntu-gcc7.5
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

### Cross-Compilation Setup
```bash
# Setup for cross-compilation (if needed)
export CROSS_COMPILE=aarch64-linux-gnu-
export CC=aarch64-linux-gnu-gcc
export CXX=aarch64-linux-gnu-g++
export AR=aarch64-linux-gnu-ar
```

## Maintenance

### Regular Updates
```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Update Python packages
pip3 install --upgrade numpy scipy matplotlib pillow

# Check for SNPE SDK updates
# Visit: https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk
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
# Monitor CPU/GPU/DSP usage
htop
cat /sys/class/kgsl/kgsl-3d0/gpubusy
cat /sys/kernel/debug/msm_subsys/slpi

# Monitor power consumption
cat /sys/class/power_supply/*/power_now

# Monitor temperatures
watch -n 1 'cat /sys/class/thermal/thermal_zone*/temp'
```

### SNPE Profiling
```bash
# Enable SNPE profiling
export SNPE_PROFILING_LEVEL=basic
snpe-net-run --profiling_level basic
```

## Support

For installation issues:
1. Check the troubleshooting section above
2. Review Qualcomm SNPE documentation: https://developer.qualcomm.com/docs/snpe/
3. Consult Qualcomm Developer Forums: https://developer.qualcomm.com/forums/
4. Check system logs: `dmesg` and `/var/log/syslog`

## References

- [Qualcomm Neural Processing SDK](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)
- [QCS6490 Technical Reference](https://www.qualcomm.com/products/internet-of-things/industrial/building-enterprise/qcs6490)
- [Robotics RB5 Development Kit](https://developer.qualcomm.com/qualcomm-robotics-rb5-kit)
- [ARM NEON Optimization Guide](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon)
