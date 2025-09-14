# Qualcomm QCS6490 Platform Guide

This directory contains complete setup, benchmarking, and analysis tools for the Qualcomm QCS6490 platform in the embedded AI benchmark suite.

## 🎯 Platform Overview

**Target Hardware**: Qualcomm QCS6490 Development Kit (Thundercomm TurboX C6490)  
**Architecture**: ARM64 with Hexagon NPU acceleration  
**AI Framework**: Qualcomm Neural Processing SDK (SNPE)  
**Key Features**: Heterogeneous compute with dedicated NPU, advanced DSP capabilities  

### Hardware Specifications
- **CPU**: 8-core Qualcomm Kryo 670 (1x A78 @ 2.7GHz, 3x A78 @ 2.4GHz, 4x A55 @ 1.9GHz)
- **GPU**: Qualcomm Adreno 643 @ 812 MHz
- **AI Accelerator**: Qualcomm Hexagon NPU with Tensor Accelerator
- **Memory**: 8GB LPDDR5 @ ~51.2 GB/s
- **Peak AI Performance**: 12-13 TOPS (INT8)
- **Power Envelope**: 5W - 12W (typical)

## 📁 Directory Structure

```
qualcomm-qcs6490/
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
- [ ] Qualcomm QCS6490 development board (Thundercomm TurboX C6490 recommended)
- [ ] Active cooling solution (40mm fan + heatsink) - **MANDATORY**
- [ ] 12V/3A DC power supply
- [ ] microSD card (128GB+) or eUFS storage
- [ ] Ubuntu 20.04 LTS (ARM64) flashed to storage
- [ ] Network connectivity for downloads
- [ ] **Qualcomm SNPE SDK** (requires manual download and registration)
- [ ] Yokogawa WT300E power analyzer (optional but recommended)

### 1. SNPE SDK Setup (MANDATORY)
**IMPORTANT**: The Qualcomm SNPE SDK must be manually downloaded before running the setup script.

```bash
# 1. Register and download SNPE SDK from Qualcomm
# Visit: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/introduction.html
# Download: SNPE SDK v2.x+ for Linux ARM64

# 2. Extract SNPE SDK
tar -xzf snpe-*.tgz
mv snpe-* ~/snpe-sdk

# 3. Source SNPE environment
cd ~/snpe-sdk/
source bin/envsetup.sh

# 4. Verify SNPE installation
echo $SNPE_ROOT
ls $SNPE_ROOT/bin/x86_64-linux-clang/snpe-onnx-to-dlc
```

### 2. Automated Platform Setup
```bash
# Clone the benchmark repository
git clone <repository-url>
cd embedded-ai-benchmark-suite/platforms/qualcomm-qcs6490/

# Run automated setup (requires SNPE SDK to be already installed)
cd setup/
chmod +x install_all.sh
./install_all.sh

# Reboot after setup completion
sudo reboot
```

### 3. Prepare Datasets
```bash
# After reboot, prepare all required datasets
cd ../../datasets/
chmod +x prepare_all_datasets.sh
./prepare_all_datasets.sh

# Follow instructions for manual KITTI and Cityscapes downloads
```

### 4. Run Individual Benchmarks
```bash
# Source the environment (includes SNPE)
source ~/benchmark_workspace/setup_env.sh
source ~/snpe-sdk/bin/envsetup.sh

# ORB-SLAM3 CPU Benchmark
cd ../platforms/qualcomm-qcs6490/orb-slam3/
./run_benchmark.sh

# 3D Object Detection Pipeline
cd ../3d-object-detection/
./run_benchmark.sh

# Semantic Segmentation
cd ../semantic-segmentation/
./run_benchmark.sh
```

### 5. Analyze Results
```bash
# Run comprehensive analysis
cd ../../analysis/scripts/
python3 analyze_all_results.py \
    --results-root ../../platforms/ \
    --output-dir ../qualcomm-analysis/
```

## 🏗️ Platform-Specific Features

### SNPE Model Optimization
All AI models are optimized using Qualcomm SNPE for maximum performance:
- **DLC Format**: Deep Learning Container optimized for Hexagon NPU
- **INT8 Quantization**: Post-training quantization for NPU acceleration
- **HTP Optimization**: Hexagon Tensor Processor specific optimizations
- **Multi-runtime Support**: CPU, GPU, DSP/HTP runtime options

### Heterogeneous Compute Strategy
The QCS6490 supports multiple AI acceleration paths:
1. **CPU Path**: Uses Kryo 670 cores for general compute
2. **GPU Path**: Uses Adreno 643 for parallel compute workloads
3. **DSP/HTP Path**: Uses Hexagon NPU for maximum AI performance and efficiency

### Power Management Integration
- **CPU Governor Control**: Performance vs power-saving modes
- **Frequency Scaling**: Dynamic voltage and frequency scaling (DVFS)
- **Hexagon NPU Power**: Dedicated power management for AI workloads

## 📊 Expected Performance Ranges

### ORB-SLAM3 (CPU Benchmark)
| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Throughput | 8-15 FPS | Kryo 670 heterogeneous CPU performance |
| P99 Latency | 80-150 ms | Depends on thermal conditions |
| Power | 8-15 W | Total system power |

### 3D Object Detection (AI Benchmark)
| Runtime | Throughput | Latency (P99) | Power | Notes |
|---------|------------|---------------|-------|-------|
| CPU | 1.4-2.7 FPS | 365-730 ms | 8-12 W | CPU-only inference |
| DSP/HTP | 2.6-5.1 FPS | 195-380 ms | 10-15 W | NPU acceleration |

### Semantic Segmentation (AI Benchmark)
| Runtime | Throughput | Latency (P99) | Power | Notes |
|---------|------------|---------------|-------|-------|
| CPU | 5-12 FPS | 80-200 ms | 8-12 W | CPU-only inference |
| DSP/HTP | 15-30 FPS | 33-67 ms | 10-15 W | NPU optimization |

## ⚙️ Configuration and Optimization

### CPU Performance Modes
```bash
# Set CPU governor to performance mode
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance | sudo tee "$cpu"
done

# Check current CPU frequencies
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq

# Monitor CPU utilization
htop
```

### SNPE Runtime Selection
```bash
# Test available SNPE runtimes
$SNPE_ROOT/bin/aarch64-android/snpe-platform-validator

# Run model on specific runtime
$SNPE_ROOT/bin/aarch64-android/snpe-net-run \
    --container model.dlc \
    --use_dsp  # or --use_cpu, --use_gpu
```

### Thermal Management
```bash
# Monitor system temperatures
cat /sys/class/thermal/thermal_zone*/temp

# Check for thermal throttling
dmesg | grep -i thermal

# Ensure active cooling is working
# CPU should stay below 85°C under load
```

## 🔧 Troubleshooting

### Common Issues

#### 1. SNPE SDK Not Found
```bash
# Error: SNPE_ROOT not set
# Solution: Download and install SNPE SDK manually
echo $SNPE_ROOT
source ~/snpe-sdk/bin/envsetup.sh

# Verify SNPE tools are available
ls $SNPE_ROOT/bin/x86_64-linux-clang/snpe-onnx-to-dlc
```

#### 2. Model Conversion Failures
```bash
# Check SNPE version compatibility
$SNPE_ROOT/bin/x86_64-linux-clang/snpe-onnx-to-dlc --help

# Verify ONNX model format
python3 -c "import onnx; model = onnx.load('model.onnx'); print('Valid ONNX model')"

# Test model conversion manually
$SNPE_ROOT/bin/x86_64-linux-clang/snpe-onnx-to-dlc \
    --input_network model.onnx \
    --output_path model.dlc
```

#### 3. Runtime Failures
```bash
# Check if DSP/HTP runtime is available
$SNPE_ROOT/bin/aarch64-android/snpe-platform-validator | grep -i dsp

# Test with CPU runtime if DSP fails
$SNPE_ROOT/bin/aarch64-android/snpe-net-run \
    --container model.dlc \
    --use_cpu
```

#### 4. Performance Issues
```bash
# Check CPU frequencies
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq

# Monitor thermal throttling
watch -n1 "cat /sys/class/thermal/thermal_zone*/temp"

# Check power supply stability
# Ensure 12V/3A minimum power supply
```

### Debug Mode
```bash
# Run benchmarks with debug output
export SNPE_LOG_LEVEL=DEBUG
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
- **CPU Frequency Locking**: Fixed performance governor settings
- **Process Isolation**: CPU affinity for benchmark processes
- **Runtime Consistency**: Same SNPE SDK version across all tests

### Measurement Accuracy
- **Hardware Power Measurement**: External Yokogawa WT300E analyzer
- **High-Resolution Timing**: ARM PMU counters and perf_counter timing
- **Statistical Significance**: Multiple runs with proper statistical analysis
- **Reproducibility**: Complete environment documentation and automation

### Qualcomm-Specific Considerations
- **Heterogeneous CPU**: Different core types with different performance characteristics
- **Hexagon NPU**: Dedicated tensor processing unit with specialized instruction set
- **SNPE Optimization**: Qualcomm's proprietary AI optimization framework
- **Memory Architecture**: LPDDR5 with optimized memory controllers

## 🚀 Advanced Usage

### Custom Model Deployment
```bash
# Convert your own ONNX model
cd ~/benchmark_workspace/models/onnx/
$SNPE_ROOT/bin/x86_64-linux-clang/snpe-onnx-to-dlc \
    --input_network your_model.onnx \
    --output_path your_model.dlc

# Quantize for NPU
$SNPE_ROOT/bin/x86_64-linux-clang/snpe-dlc-quantize \
    --input_dlc your_model.dlc \
    --output_dlc your_model_quantized.dlc \
    --enable_htp
```

### Profiling and Optimization
```bash
# Enable SNPE profiling
export SNPE_PROFILE_LEVEL=detailed

# Run with profiling
./run_benchmark.sh

# Analyze profiling results
$SNPE_ROOT/bin/x86_64-linux-clang/snpe-diagview \
    --input_log profile.log
```

### Multi-Runtime Comparison
```bash
# Run same model on different runtimes
./run_benchmark.sh --runtime cpu
./run_benchmark.sh --runtime dsp
./run_benchmark.sh --runtime gpu
```

## 📊 Performance Comparison

### vs. NVIDIA Jetson Orin NX
- **CPU**: Kryo 670 vs ARM Cortex-A78AE
- **AI Acceleration**: Hexagon NPU vs GPU/DLA
- **Memory**: LPDDR5 51.2 GB/s vs 102.4 GB/s
- **Software**: SNPE vs TensorRT

### vs. Intel N100
- **Architecture**: ARM64 vs x86_64
- **AI Acceleration**: Dedicated NPU vs Integrated Graphics
- **Power**: 5-12W vs 6-15W
- **Software**: SNPE vs OpenVINO

## 🤝 Contributing

### Adding New Benchmarks
1. Create benchmark directory under `qualcomm-qcs6490/`
2. Implement SNPE optimization pipeline
3. Add CPU, GPU, and DSP/HTP runtime support
4. Include comprehensive README with setup instructions
5. Test on actual QCS6490 hardware

### Optimization Guidelines
- Always implement multiple runtime paths (CPU, DSP/HTP, GPU)
- Use INT8 quantization for fair comparison with other platforms
- Include proper thermal and power management
- Document SNPE SDK version requirements
- Provide fallback options for different hardware configurations

## ⚠️ Important Notes

### SNPE SDK Licensing
- The Qualcomm SNPE SDK requires registration and agreement to license terms
- SDK must be manually downloaded from Qualcomm's developer portal
- Commercial use may require additional licensing agreements

### Hardware Availability
- QCS6490 development boards may have limited availability
- Alternative boards with QCS6490 SoC can be used but may require setup modifications
- Some features may require specific board configurations

### Performance Variability
- Performance highly dependent on thermal design and cooling solution
- SNPE SDK version can significantly impact performance
- Different board designs may have different power delivery and thermal characteristics

---

**Note**: This platform guide assumes Qualcomm SNPE SDK v2.x+ and Ubuntu 20.04 LTS ARM64. Performance results are highly dependent on thermal conditions, power supply stability, and SNPE SDK version. The SNPE SDK must be manually downloaded and installed before running the setup script. Always report SNPE SDK version, board model, and environmental factors with benchmark results.
