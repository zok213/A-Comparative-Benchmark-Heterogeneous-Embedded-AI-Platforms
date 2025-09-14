# NVIDIA Jetson Orin NX Platform Guide

This directory contains complete setup, benchmarking, and analysis tools for the NVIDIA Jetson Orin NX platform in the embedded AI benchmark suite.

## 🎯 Platform Overview

**Target Hardware**: NVIDIA Jetson Orin NX Developer Kit  
**Architecture**: ARM64 with GPU acceleration  
**AI Framework**: TensorRT with CUDA  
**Key Features**: GPU/DLA dual acceleration, unified memory architecture  

### Hardware Specifications
- **CPU**: 6-core ARM Cortex-A78AE @ 2.0 GHz
- **GPU**: 1024-core NVIDIA Ampere architecture with 32 Tensor Cores
- **AI Accelerator**: 1x NVDLA v2.0 (Deep Learning Accelerator)
- **Memory**: 8GB or 16GB 128-bit LPDDR5 @ 102.4 GB/s
- **Peak AI Performance**: 70 TOPS (sparse INT8)
- **Power Envelope**: 10W - 20W (configurable with nvpmodel)

## 📁 Directory Structure

```
nvidia-jetson/
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
- [ ] NVIDIA Jetson Orin NX Developer Kit (16GB recommended)
- [ ] Active cooling solution (fan + heatsink) - **MANDATORY**
- [ ] 19V/65W power supply
- [ ] microSD card (128GB+) or NVMe SSD
- [ ] Ubuntu 20.04 LTS flashed to storage
- [ ] Network connectivity for downloads
- [ ] Yokogawa WT300E power analyzer (optional but recommended)

### 1. Initial Setup
```bash
# Clone the benchmark repository
git clone <repository-url>
cd embedded-ai-benchmark-suite/platforms/nvidia-jetson/

# Run automated setup (this will take 30-60 minutes)
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
cd ../platforms/nvidia-jetson/orb-slam3/
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
    --output-dir ../nvidia-jetson-analysis/
```

## 🏗️ Platform-Specific Features

### TensorRT Optimization
All AI models are optimized using TensorRT for maximum performance:
- **INT8 Quantization**: Post-training quantization for 4x speedup
- **GPU Optimization**: Optimized for Ampere architecture
- **DLA Optimization**: Offloading to dedicated Deep Learning Accelerator
- **Dynamic Shapes**: Support for variable input sizes

### Dual Acceleration Strategy
The Jetson Orin NX supports two AI acceleration paths:
1. **GPU Path**: Uses CUDA cores and Tensor Cores for maximum throughput
2. **DLA Path**: Uses dedicated Deep Learning Accelerator for power efficiency

### Power Management Integration
- **nvpmodel**: Configurable power modes (MAXN, 15W, 10W)
- **jetson_clocks**: Lock frequencies for consistent benchmarking
- **tegrastats**: Built-in power monitoring (supplemented by external measurement)

## 📊 Expected Performance Ranges

### ORB-SLAM3 (CPU Benchmark)
| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Throughput | 20-35 FPS | ARM Cortex-A78AE performance |
| P99 Latency | 35-60 ms | Depends on thermal conditions |
| Power | 12-18 W | Total system power |

### 3D Object Detection (AI Benchmark)
| Runtime | Throughput | Latency (P99) | Power | Notes |
|---------|------------|---------------|-------|-------|
| GPU | 8-15 FPS | 80-150 ms | 15-20 W | Maximum performance |
| DLA | 5-10 FPS | 120-250 ms | 10-15 W | Power efficient |

### Semantic Segmentation (AI Benchmark)
| Runtime | Throughput | Latency (P99) | Power | Notes |
|---------|------------|---------------|-------|-------|
| GPU | 25-45 FPS | 25-50 ms | 15-20 W | Real-time capable |
| DLA | 15-30 FPS | 40-80 ms | 10-15 W | Balanced performance |

## ⚙️ Configuration and Optimization

### Power Modes
```bash
# Set maximum performance mode
sudo nvpmodel -m 0  # MAXN mode

# Set power-efficient mode
sudo nvpmodel -m 2  # 15W mode

# Lock clocks for benchmarking
sudo jetson_clocks
```

### Thermal Management
```bash
# Monitor temperatures
tegrastats

# Check thermal zones
cat /sys/class/thermal/thermal_zone*/temp

# Ensure active cooling is working
# CPU should stay below 85°C under load
```

### Memory Optimization
```bash
# Check memory usage
free -h

# Increase swap if needed (for model conversion)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## 🔧 Troubleshooting

### Common Issues

#### 1. JetPack Installation Issues
```bash
# Verify JetPack version
jetson_release

# Check CUDA installation
nvcc --version

# Verify TensorRT
ls /usr/src/tensorrt/bin/trtexec
```

#### 2. Model Conversion Failures
```bash
# Check TensorRT version compatibility
/usr/src/tensorrt/bin/trtexec --help

# Verify ONNX model format
python3 -c "import onnx; model = onnx.load('model.onnx'); print('Valid ONNX model')"

# Check calibration data
ls ~/benchmark_workspace/results/*/calibration/
```

#### 3. Performance Issues
```bash
# Check if thermal throttling is occurring
tegrastats | grep -E "(CPU|GPU|temp)"

# Verify power mode
cat /etc/nvpmodel.conf
nvpmodel -q

# Check clock frequencies
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
cat /sys/kernel/debug/clk/gpc0/clk_rate
```

#### 4. Memory Issues
```bash
# Monitor memory during benchmark
watch -n1 free -h

# Check for memory leaks
valgrind --tool=memcheck --leak-check=yes ./benchmark_app

# Increase virtual memory
sudo sysctl vm.overcommit_memory=1
```

### Debug Mode
```bash
# Run benchmarks with debug output
export CUDA_LAUNCH_BLOCKING=1
export TRT_LOGGER_VERBOSITY=VERBOSE
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
- **Power Mode Consistency**: Fixed nvpmodel settings across runs
- **Clock Locking**: jetson_clocks ensures consistent frequencies
- **Process Isolation**: CPU affinity and nice levels for benchmark processes

### Measurement Accuracy
- **Hardware Power Measurement**: External Yokogawa WT300E analyzer
- **High-Resolution Timing**: CUDA events for GPU timing, CPU perf_counter for CPU
- **Statistical Significance**: Multiple runs with proper statistical analysis
- **Reproducibility**: Complete environment documentation and automation

### NVIDIA-Specific Considerations
- **Unified Memory Architecture**: CPU and GPU share LPDDR5 memory
- **Tensor Core Utilization**: INT8 operations leverage specialized hardware
- **DLA Offloading**: Power-efficient inference path
- **CUDA Compute Capability**: 8.7 (Ampere architecture)

## 🚀 Advanced Usage

### Custom Model Deployment
```bash
# Convert your own ONNX model
cd ~/benchmark_workspace/models/onnx/
/usr/src/tensorrt/bin/trtexec \
    --onnx=your_model.onnx \
    --saveEngine=your_model.engine \
    --int8 \
    --workspace=4096
```

### Profiling and Optimization
```bash
# Profile CUDA kernels
nsys profile --stats=true ./benchmark_app

# Analyze TensorRT engine
/usr/src/tensorrt/bin/trtexec \
    --loadEngine=model.engine \
    --profilingVerbosity=detailed
```

### Multi-Model Inference
```bash
# Run concurrent inference on GPU and DLA
./run_benchmark.sh --concurrent-gpu-dla
```

## 📊 Performance Comparison

### vs. Qualcomm QCS6490
- **CPU**: ARM Cortex-A78AE vs Kryo 670
- **AI Acceleration**: GPU/DLA vs Hexagon NPU
- **Memory**: LPDDR5 102.4 GB/s vs ~51.2 GB/s
- **Software**: TensorRT vs SNPE

### vs. Intel N100
- **Architecture**: ARM64 vs x86_64
- **AI Acceleration**: Dedicated GPU/DLA vs Integrated UHD Graphics
- **Power**: 10-20W vs 6-15W
- **Software**: TensorRT vs OpenVINO

## 🤝 Contributing

### Adding New Benchmarks
1. Create benchmark directory under `nvidia-jetson/`
2. Implement TensorRT optimization pipeline
3. Add GPU and DLA runtime support
4. Include comprehensive README with setup instructions
5. Test on actual Jetson Orin NX hardware

### Optimization Guidelines
- Always implement both GPU and DLA paths where applicable
- Use INT8 quantization for fair comparison with other platforms
- Include proper thermal and power management
- Document CUDA compute capability requirements
- Provide fallback options for different JetPack versions

---

**Note**: This platform guide assumes NVIDIA JetPack 5.1.1+ with TensorRT 8.5.2+. Performance results are highly dependent on thermal conditions, power mode settings, and JetPack version. Always report these environmental factors with benchmark results.
