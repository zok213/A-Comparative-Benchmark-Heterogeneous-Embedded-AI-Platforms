# Radxa CM5 (RK3588S) Platform Guide

This directory contains complete setup, benchmarking, and analysis tools for the Radxa CM5 platform with Rockchip RK3588S processor in the embedded AI benchmark suite.

## 🎯 Platform Overview

**Target Hardware**: Radxa CM5 Compute Module with RK3588S  
**Architecture**: ARM64 with dedicated NPU acceleration  
**AI Framework**: RKNN Toolkit with OpenVINO support  
**Key Features**: High-performance ARM computing, dedicated 6 TOPS NPU, Mali GPU acceleration  

### Hardware Specifications
- **CPU**: 8-core ARM (4x Cortex-A76 @ 2.4 GHz + 4x Cortex-A55 @ 1.8 GHz)
- **GPU**: Mali-G610 MP4 GPU with OpenGL ES 3.2, OpenCL 2.2, Vulkan 1.2
- **AI Accelerator**: NPU with 6 TOPS INT8 performance
- **Memory**: 4GB/8GB/16GB/32GB LPDDR4/LPDDR4x/LPDDR5
- **Peak AI Performance**: 6 TOPS (INT8)
- **Power Envelope**: 5-15W (configurable)

## 📁 Directory Structure

```
radxa-cm5/
├── 📄 README.md                    # This comprehensive platform guide
├── 📁 setup/                       # Platform installation and configuration
│   ├── 📄 install_all.sh          # Automated setup script
│   └── 📄 README.md               # Setup instructions and troubleshooting
├── 📁 orb-slam3/                   # ORB-SLAM3 CPU benchmark
│   ├── 📄 run_benchmark.sh        # Benchmark execution script
│   └── 📄 README.md               # Detailed benchmark guide
├── 📁 3d-object-detection/         # 3D Object Detection pipeline
│   ├── 📄 run_benchmark.sh        # Benchmark execution script
│   └── 📄 README.md               # Detailed benchmark guide
└── 📁 semantic-segmentation/       # Semantic Segmentation benchmark
    ├── 📄 run_benchmark.sh        # Benchmark execution script
    └── 📄 README.md               # Detailed benchmark guide
```

## 🚀 Quick Start

### Prerequisites Checklist
- [ ] Radxa CM5 compute module with RK3588S
- [ ] 16GB LPDDR5 RAM (recommended) or 8GB minimum
- [ ] Active cooling solution (heatsink + fan) - **MANDATORY**
- [ ] 12V/2A DC power supply or USB-C PD (24W minimum)
- [ ] microSD card (128GB+) or eMMC storage
- [ ] Ubuntu 20.04 LTS (ARM64) or Debian 11 flashed to storage
- [ ] Network connectivity for downloads
- [ ] Yokogawa WT300E power analyzer (optional but recommended)

### 1. Initial Setup
```bash
# Clone the benchmark repository
git clone <repository-url>
cd embedded-ai-benchmark-suite/platforms/radxa-cm5/

# Run automated setup (this will take 30-45 minutes)
cd setup/
chmod +x install_all.sh
./install_all.sh

# Reboot after setup completion
sudo reboot
```

### 2. Prepare Datasets
```bash
# After reboot, prepare all required datasets
cd ../../datasets/
chmod +x prepare_all_datasets.sh
./prepare_all_datasets.sh

# Follow instructions for manual KITTI and Cityscapes downloads
```

### 3. Run Individual Benchmarks
```bash
# Source the environment
source ~/benchmark_workspace/setup_env.sh

# ORB-SLAM3 CPU Benchmark
cd ../platforms/radxa-cm5/orb-slam3/
./run_benchmark.sh

# 3D Object Detection Pipeline
cd ../3d-object-detection/
./run_benchmark.sh

# Semantic Segmentation
cd ../semantic-segmentation/
./run_benchmark.sh
```

### 4. Analyze Results
```bash
# Run comprehensive analysis
cd ../../analysis/scripts/
python3 analyze_all_results.py \
    --results-root ../../platforms/ \
    --output-dir ../radxa-cm5-analysis/
```

## 🏗️ Platform-Specific Features

### RKNN Optimization
All AI models are optimized using RKNN toolkit for maximum performance:
- **RKNN Converter**: Converts ONNX/TensorFlow/PyTorch to RKNN format
- **Post-Training Quantization**: INT8 quantization for NPU acceleration
- **NPU Optimization**: Dedicated 6 TOPS neural processing unit
- **Mali GPU Support**: GPU acceleration for parallel computing tasks
- **ARM CPU Optimization**: Optimized for big.LITTLE Cortex-A76/A55 cores

### Multi-Compute Strategy
The RK3588S platform supports three main acceleration paths:
1. **NPU Path**: Uses dedicated 6 TOPS NPU for AI inference
2. **GPU Path**: Uses Mali-G610 MP4 GPU for parallel computing
3. **CPU Path**: Uses ARM big.LITTLE cores for general processing

### Power Management Integration
- **ARM DVFS**: Dynamic voltage and frequency scaling for CPU clusters
- **Mali GPU Scaling**: Dynamic GPU frequency and voltage control
- **NPU Power Management**: Intelligent NPU power gating and scaling
- **Thermal Management**: Advanced thermal throttling and monitoring

## 📊 Expected Performance Ranges

### ORB-SLAM3 (CPU Benchmark)
| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Throughput | 25-40 FPS | ARM Cortex-A76 high-performance cores |
| P99 Latency | 30-50 ms | Superior ARM architecture performance |
| Power | 6-10 W | Total system power including peripherals |

### 3D Object Detection (AI Benchmark)
| Runtime | Throughput | Latency (P99) | Power | Notes |
|---------|------------|---------------|-------|-------|
| NPU | 8-15 FPS | 67-125 ms | 8-12 W | NPU accelerated inference |
| GPU | 4-8 FPS | 125-250 ms | 10-15 W | Mali GPU acceleration |
| CPU | 1.5-3 FPS | 333-667 ms | 6-10 W | ARM CPU-only inference |

### Semantic Segmentation (AI Benchmark)
| Runtime | Throughput | Latency (P99) | Power | Notes |
|---------|------------|---------------|-------|-------|
| NPU | 45-80 FPS | 12-22 ms | 8-12 W | NPU accelerated inference |
| GPU | 25-45 FPS | 22-40 ms | 10-15 W | Mali GPU acceleration |
| CPU | 12-25 FPS | 40-83 ms | 6-10 W | ARM CPU-only inference |

## ⚙️ Configuration and Optimization

### CPU Performance Modes
```bash
# Set CPU governor to performance mode for both clusters
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Set maximum frequencies for big cores (A76)
echo 2400000 | sudo tee /sys/devices/system/cpu/cpu[4-7]/cpufreq/scaling_max_freq

# Set maximum frequencies for LITTLE cores (A55)
echo 1800000 | sudo tee /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_max_freq

# Check current CPU frequencies
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
```

### Mali GPU Configuration
```bash
# Check GPU availability
ls /sys/class/misc/mali0/
cat /sys/kernel/debug/mali0/ctx

# Monitor GPU frequency
cat /sys/class/devfreq/fb000000.gpu/cur_freq

# Set GPU governor to performance
echo performance | sudo tee /sys/class/devfreq/fb000000.gpu/governor

# Set GPU to maximum frequency
cat /sys/class/devfreq/fb000000.gpu/max_freq | sudo tee /sys/class/devfreq/fb000000.gpu/min_freq
```

### RKNN Runtime Selection
```bash
# Test available RKNN runtime
python3 -c "
from rknn.api import RKNN
rknn = RKNN()
print('RKNN toolkit version:', rknn.get_version())
print('Available targets: NPU, GPU, CPU')
"

# Run model on specific device
# Use RKNN runtime for inference on NPU, GPU, or CPU
```

### Thermal Management
```bash
# Monitor CPU temperatures
sensors | grep Core

# Check for thermal throttling
dmesg | grep -i thermal

# Monitor frequencies during load
watch -n1 "cat /proc/cpuinfo | grep MHz"
```

## 🔧 Troubleshooting

### Common Issues

#### 1. OpenVINO Installation Issues
```bash
# Verify OpenVINO installation
python3 -c "import openvino; print('OpenVINO version:', openvino.__version__)"

# Check if GPU plugin is available
python3 -c "
from openvino.runtime import Core
core = Core()
devices = core.available_devices
print('GPU available:', 'GPU' in devices)
"

# Reinstall if needed
pip install --upgrade openvino openvino-dev
```

#### 2. Intel GPU Driver Issues
```bash
# Check GPU driver status
lspci -k | grep -A 3 VGA

# Install Intel GPU drivers
sudo apt update
sudo apt install intel-gpu-tools

# Verify GPU functionality
intel_gpu_top
```

#### 3. Model Conversion Failures
```bash
# Check OpenVINO Model Optimizer
mo --help

# Test model conversion manually
mo --input_model model.onnx --output_dir output/

# Verify converted model
python3 -c "
from openvino.runtime import Core
core = Core()
model = core.read_model('model.xml')
print('Model loaded successfully')
"
```

#### 4. Performance Issues
```bash
# Check if thermal throttling is occurring
sensors | grep -E "(Core|Package)"

# Verify turbo boost is working
cat /sys/devices/system/cpu/intel_pstate/no_turbo  # Should be 0

# Check power supply adequacy
# Ensure USB-C PD 45W minimum

# Monitor system load
htop
```

### Debug Mode
```bash
# Run benchmarks with debug output
export OV_CPU_PROFILE=1
export OV_GPU_PROFILE=1
bash -x run_benchmark.sh
```

## 📖 Detailed Guides

### Individual Benchmark Guides
- [ORB-SLAM3 Setup and Execution](orb-slam3/README.md)
- [3D Object Detection Pipeline](3d-object-detection/README.md)
- [Semantic Segmentation Benchmark](semantic-segmentation/README.md)

### Technical Documentation
- [Setup and Installation Guide](setup/README.md)
- [Hardware Requirements](../../docs/hardware-requirements.md)
- [Power Measurement Setup](../../docs/power-measurement-setup.md)

## 🔬 Scientific Methodology

### Experimental Controls
- **Thermal Stability**: Active cooling prevents throttling
- **CPU Frequency Control**: Performance governor for consistent clocks
- **Process Isolation**: CPU affinity for benchmark processes
- **Turbo Boost Management**: Consistent turbo behavior across runs

### Measurement Accuracy
- **Hardware Power Measurement**: External power analyzer (USB-C PD measurement)
- **High-Resolution Timing**: Intel TSC and perf_counter timing
- **Statistical Significance**: Multiple runs with proper statistical analysis
- **Reproducibility**: Complete environment documentation and automation

### Intel-Specific Considerations
- **x86_64 Architecture**: Different instruction set from ARM platforms
- **Shared Memory**: CPU and GPU share LPDDR5 memory
- **Turbo Boost Variability**: Dynamic frequency scaling affects consistency
- **TDP Constraints**: 6W TDP may limit sustained performance

## 🚀 Advanced Usage

### Custom Model Deployment
```bash
# Convert your own ONNX model
cd ~/benchmark_workspace/models/onnx/
mo --input_model your_model.onnx --output_dir ../openvino/your_model/

# Quantize model for better performance
pot -c quantization_config.json --output-dir your_model_int8/
```

### Profiling and Optimization
```bash
# Profile OpenVINO inference
benchmark_app -m model.xml -d CPU -report_type detailed_counters

# Analyze performance bottlenecks
python3 -c "
import openvino.runtime as ov
# Add profiling code here
"
```

### Multi-Device Inference
```bash
# Run inference on multiple devices simultaneously
benchmark_app -m model.xml -d MULTI:CPU,GPU
```

### Memory Usage Optimization
```bash
# Monitor memory usage
sudo iotop -a

# Optimize for low memory systems
export OV_CPU_THREADS_NUM=2
export OV_GPU_CACHE_CAPACITY=0
```

## 📊 Performance Comparison

### vs. NVIDIA Jetson Orin NX
- **Architecture**: ARM64 vs ARM64 (similar)
- **AI Acceleration**: 6 TOPS NPU vs 70 TOPS GPU/DLA
- **Memory**: LPDDR4x/5 vs LPDDR5 unified
- **Software**: RKNN vs TensorRT

### vs. Qualcomm QCS6490
- **Architecture**: ARM64 vs ARM64 (similar)
- **AI Acceleration**: 6 TOPS NPU vs 12-13 TOPS Hexagon NPU
- **CPU Performance**: Cortex-A76/A55 vs Kryo 670
- **Software**: RKNN vs SNPE

### Unique RK3588S Advantages
- **NPU Acceleration**: Dedicated 6 TOPS neural processing unit
- **High CPU Performance**: Powerful Cortex-A76 cores at 2.4 GHz
- **Mali GPU**: Advanced Mali-G610 MP4 GPU with Vulkan 1.2
- **Power Efficiency**: Excellent performance per watt ratio

## 🤝 Contributing

### Adding New Benchmarks
1. Create benchmark directory under `radxa-cm5/`
2. Implement RKNN optimization pipeline
3. Add CPU and GPU runtime support
4. Include comprehensive README with setup instructions
5. Test on actual Radxa CM5 hardware

### Optimization Guidelines
- Always implement NPU, GPU, and CPU paths where applicable
- Use RKNN converter for all model optimizations
- Include proper thermal and power management
- Document RKNN toolkit version requirements
- Provide fallback options for NPU-less systems

### RK3588S-Specific Considerations
- Test with different Mali GPU driver versions
- Validate performance across different power profiles
- Consider big.LITTLE scheduling behavior in benchmark design
- Account for NPU memory bandwidth limitations

## ⚠️ Important Notes

### Hardware Limitations
- **NPU Memory**: Limited NPU memory may affect large model deployment
- **Shared Memory**: CPU, GPU, and NPU compete for memory bandwidth
- **Thermal Throttling**: High-performance cores may throttle under sustained load
- **Cooling Dependency**: Performance highly dependent on thermal solution

### Software Dependencies
- **RKNN Toolkit Version**: Performance varies significantly across versions
- **Mali GPU Drivers**: Linux Mali driver maturity affects stability
- **OS Compatibility**: Best performance with Ubuntu 20.04 LTS (ARM64) or Debian 11
- **Python Dependencies**: Specific RKNN package versions may be required

### Power Measurement Challenges
- **DC Power**: Requires current shunt for accurate measurement
- **Dynamic Power**: Big.LITTLE and NPU create variable power consumption
- **System vs SoC**: Total system power includes peripherals and converters

---

**Note**: This platform guide assumes RKNN Toolkit 1.5.x+ and Ubuntu 20.04 LTS ARM64. Performance results are highly dependent on thermal conditions, power delivery, big.LITTLE scheduling, and RKNN toolkit version. The RK3588S design balances high performance with power efficiency through its dedicated NPU and advanced power management. Always report cooling solution, ambient temperature, power supply specifications, and software versions with benchmark results.
