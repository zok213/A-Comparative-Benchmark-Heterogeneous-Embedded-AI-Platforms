# Radxa X4 (Intel N100) Platform Guide

This directory contains complete setup, benchmarking, and analysis tools for the Radxa X4 platform with Intel N100 processor in the embedded AI benchmark suite.

## 🎯 Platform Overview

**Target Hardware**: Radxa X4 Single-Board Computer with Intel N100  
**Architecture**: x86_64 with integrated graphics acceleration  
**AI Framework**: Intel OpenVINO Toolkit  
**Key Features**: x86 embedded computing, Intel UHD Graphics acceleration, unified memory architecture  

### Hardware Specifications
- **CPU**: 4-core Intel N100 (Alder Lake-N) @ 1.0 GHz base, 3.4 GHz turbo
- **GPU**: Intel UHD Graphics (24 Execution Units)
- **AI Accelerator**: Intel GNA 3.0 (Gaussian & Neural Accelerator)
- **Memory**: 8GB or 16GB LPDDR5 @ up to 4800 MT/s
- **Peak AI Performance**: Varies by workload (no published TOPS rating)
- **Power Envelope**: 6W TDP (configurable)

## 📁 Directory Structure

```
radxa-x4/
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
- [ ] Radxa X4 single-board computer with Intel N100
- [ ] 16GB LPDDR5 RAM (recommended) or 8GB minimum
- [ ] Active cooling solution (heatsink + fan) - **MANDATORY**
- [ ] USB-C PD power supply (45W minimum)
- [ ] microSD card (128GB+) or eMMC storage
- [ ] Ubuntu 20.04 LTS (x86_64) flashed to storage
- [ ] Network connectivity for downloads
- [ ] Yokogawa WT300E power analyzer (optional but recommended)

### 1. Initial Setup
```bash
# Clone the benchmark repository
git clone <repository-url>
cd embedded-ai-benchmark-suite/platforms/radxa-x4/

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
cd ../platforms/radxa-x4/orb-slam3/
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
    --output-dir ../radxa-x4-analysis/
```

## 🏗️ Platform-Specific Features

### OpenVINO Optimization
All AI models are optimized using Intel OpenVINO for maximum performance:
- **Model Optimizer (MO)**: Converts ONNX to OpenVINO IR format
- **Post-Training Optimization Tool (POT)**: INT8 quantization
- **CPU Optimization**: Intel CPU-specific optimizations (AVX, SSE)
- **GPU Optimization**: Intel UHD Graphics acceleration
- **Neural Network Compression Framework (NNCF)**: Advanced quantization

### Dual Compute Strategy
The Intel N100 platform supports two main acceleration paths:
1. **CPU Path**: Uses Intel N100 cores with AVX/SSE optimizations
2. **GPU Path**: Uses Intel UHD Graphics with OpenCL acceleration

### Power Management Integration
- **Intel P-State Driver**: Advanced CPU frequency scaling
- **Turbo Boost**: Dynamic frequency scaling up to 3.4 GHz
- **GPU Frequency Control**: Dynamic GPU clock scaling
- **TDP Configuration**: 6W TDP with burst capability

## 📊 Expected Performance Ranges

### ORB-SLAM3 (CPU Benchmark)
| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Throughput | 15-25 FPS | Intel N100 4-core performance |
| P99 Latency | 50-80 ms | Depends on turbo boost behavior |
| Power | 6-15 W | Total system power including peripherals |

### 3D Object Detection (AI Benchmark)
| Runtime | Throughput | Latency (P99) | Power | Notes |
|---------|------------|---------------|-------|-------|
| CPU | 0.96-1.92 FPS | 520-1040 ms | 8-12 W | CPU-only inference |
| GPU | 1.45-3.7 FPS | 270-690 ms | 10-15 W | Intel UHD Graphics acceleration |

### Semantic Segmentation (AI Benchmark)
| Runtime | Throughput | Latency (P99) | Power | Notes |
|---------|------------|---------------|-------|-------|
| CPU | 8-18 FPS | 55-125 ms | 8-12 W | CPU-only inference |
| GPU | 15-35 FPS | 28-67 ms | 10-15 W | GPU acceleration benefit |

## ⚙️ Configuration and Optimization

### CPU Performance Modes
```bash
# Set CPU governor to performance mode
sudo cpupower frequency-set -g performance

# Check current CPU frequencies
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq

# Monitor turbo boost behavior
sudo turbostat --interval 1
```

### Intel GPU Configuration
```bash
# Check GPU availability
lspci | grep VGA
ls /sys/class/drm/card0/

# Monitor GPU frequency
cat /sys/class/drm/card0/gt_cur_freq_mhz

# Set GPU to maximum performance
echo $(cat /sys/class/drm/card0/gt_max_freq_mhz) | sudo tee /sys/class/drm/card0/gt_min_freq_mhz
```

### OpenVINO Runtime Selection
```bash
# Test available OpenVINO devices
python3 -c "
from openvino.runtime import Core
core = Core()
print('Available devices:', core.available_devices)
"

# Run model on specific device
benchmark_app -m model.xml -d CPU  # or GPU, AUTO
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
- **Architecture**: x86_64 vs ARM64
- **AI Acceleration**: Intel UHD Graphics vs NVIDIA GPU/DLA
- **Memory**: LPDDR5 shared vs LPDDR5 unified
- **Software**: OpenVINO vs TensorRT

### vs. Qualcomm QCS6490
- **Architecture**: x86_64 vs ARM64
- **AI Acceleration**: Intel UHD Graphics vs Hexagon NPU
- **Power**: 6-15W vs 5-12W
- **Software**: OpenVINO vs SNPE

### Unique x86 Advantages
- **Software Compatibility**: Broader x86 software ecosystem
- **Development Tools**: Native x86 development and debugging
- **Virtualization**: Better virtualization support
- **Legacy Support**: Compatibility with x86 applications

## 🤝 Contributing

### Adding New Benchmarks
1. Create benchmark directory under `radxa-x4/`
2. Implement OpenVINO optimization pipeline
3. Add CPU and GPU runtime support
4. Include comprehensive README with setup instructions
5. Test on actual Radxa X4 hardware

### Optimization Guidelines
- Always implement both CPU and GPU paths where applicable
- Use OpenVINO Model Optimizer for all model conversions
- Include proper thermal and power management
- Document OpenVINO version requirements
- Provide fallback options for GPU-less systems

### Intel-Specific Considerations
- Test with different Intel GPU driver versions
- Validate performance across different TDP settings
- Consider turbo boost behavior in benchmark design
- Account for shared memory architecture effects

## ⚠️ Important Notes

### Hardware Limitations
- **6W TDP**: May limit sustained performance under heavy loads
- **Shared Memory**: CPU and GPU compete for memory bandwidth
- **Single Channel**: Memory architecture may limit memory-intensive workloads
- **Cooling Dependency**: Performance highly dependent on thermal solution

### Software Dependencies
- **OpenVINO Version**: Performance varies significantly across versions
- **Intel GPU Drivers**: Linux GPU driver maturity affects stability
- **Ubuntu Compatibility**: Best performance with Ubuntu 20.04 LTS
- **Python Dependencies**: Specific package versions may be required

### Power Measurement Challenges
- **USB-C PD**: Requires specialized equipment for accurate measurement
- **Dynamic Power**: Turbo boost creates variable power consumption
- **System vs SoC**: Total system power includes peripherals and converters

---

**Note**: This platform guide assumes Intel OpenVINO Toolkit 2023.x+ and Ubuntu 20.04 LTS x86_64. Performance results are highly dependent on thermal conditions, power delivery, turbo boost behavior, and OpenVINO version. The Intel N100's 6W TDP design prioritizes power efficiency over peak performance. Always report cooling solution, ambient temperature, power supply specifications, and software versions with benchmark results.
