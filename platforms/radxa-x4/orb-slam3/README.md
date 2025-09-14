# ORB-SLAM3 Benchmark - Radxa X4 (Intel N100)

This directory contains the complete setup and execution scripts for benchmarking ORB-SLAM3 performance on the Radxa X4 platform with Intel N100 processor.

## 📋 Overview

**Benchmark Type**: CPU Performance Evaluation  
**Target Platform**: Radxa X4 with Intel N100 (Alder Lake-N)  
**Workload**: Visual-Inertial SLAM using ORB-SLAM3  
**Dataset**: EuRoC MAV Machine Hall 01 sequence  
**Key Metrics**: Throughput (FPS), P99 Latency, Power Consumption  

## 🎯 Benchmark Objective

This benchmark evaluates the CPU and memory subsystem performance of the Intel N100 processor using ORB-SLAM3, a computationally intensive visual-inertial SLAM algorithm. The Intel N100 provides an interesting x86 comparison point against ARM-based embedded processors:

### Why ORB-SLAM3?
- **Multi-threaded workload**: Utilizes all 4 CPU cores simultaneously
- **Memory intensive**: Tests memory bandwidth and cache hierarchy performance
- **Real-world application**: Represents actual robotics/AR/VR workloads
- **CPU-bound**: Does not rely on specialized AI accelerators
- **Cross-architecture comparison**: Enables x86 vs ARM performance analysis

### Intel N100 Architecture
- **CPU Cores**: 4x Intel cores (no E-cores/P-cores split)
- **Base Clock**: 1.0 GHz, Turbo up to 3.4 GHz
- **Cache**: 6MB L3 cache
- **Memory**: DDR4-3200/DDR5-4800 support
- **TDP**: 6W (configurable)

## 🛠️ Prerequisites

### Hardware Requirements
- Radxa X4 single-board computer with Intel N100
- 16GB LPDDR5 RAM (recommended) or 8GB minimum
- Active cooling solution (heatsink + fan) - **MANDATORY**
- USB-C PD power supply (45W minimum)
- microSD card (64GB+) or eMMC storage
- Yokogawa WT300E power analyzer (for accurate power measurement)

### Software Requirements
- Ubuntu 20.04 LTS (x86_64)
- Platform setup completed (run `../setup/install_all.sh` first)
- EuRoC MAV dataset downloaded (run `../../datasets/prepare_all_datasets.sh`)

## 🚀 Quick Start

### 1. Verify Prerequisites
```bash
# Check CPU information
lscpu | grep -E "(Model name|CPU\(s\)|Thread|MHz)"

# Check if ORB-SLAM3 is built
ls ~/ORB_SLAM3/Examples/Monocular-Inertial/mono_inertial_euroc

# Check if dataset is available
ls ~/benchmark_workspace/datasets/euroc/MH01/

# Check environment variables
source ~/benchmark_workspace/setup_env.sh
echo $ORB_SLAM3_ROOT
echo $DATASETS_ROOT
```

### 2. Run Benchmark
```bash
cd radxa-x4/orb-slam3/
chmod +x run_benchmark.sh
./run_benchmark.sh
```

### 3. View Results
```bash
# View summary
cat ~/benchmark_workspace/results/orb_slam3/detailed_analysis.txt

# View visualization
xdg-open ~/benchmark_workspace/results/orb_slam3/performance_analysis.png
```

## 📊 Understanding Results

### Output Files
```
~/benchmark_workspace/results/orb_slam3/
├── summary.txt                       # Basic run summary
├── detailed_analysis.txt             # Comprehensive performance metrics
├── performance_analysis.png          # Performance visualization
├── logs/                             # Individual run logs
│   ├── run_1_YYYYMMDD_HHMMSS.log
│   ├── run_2_YYYYMMDD_HHMMSS.log
│   └── ...
├── system_monitor_*.log              # System monitoring data
├── freq_monitor_*.log                # CPU/GPU frequency monitoring
└── temp_monitor_*.log                # Temperature monitoring
```

### Key Metrics Explained

#### Performance Metrics
- **Throughput (FPS)**: Average frames processed per second
  - Higher values indicate better performance
  - Typical range for Intel N100: 15-30 FPS depending on thermal conditions
  
- **P99 Latency (ms)**: 99th percentile frame processing time
  - Represents worst-case performance for 99% of frames
  - Critical for real-time applications
  - Lower values are better

- **Mean Latency (ms)**: Average frame processing time
  - Overall processing speed indicator
  - Should be consistent across runs

#### System Metrics
- **CPU Utilization**: Multi-core CPU usage during benchmark
- **CPU Frequencies**: Real-time CPU clock speeds
- **GPU Frequencies**: Intel UHD Graphics clock speeds (monitored but not used)
- **Temperature**: CPU temperature monitoring for thermal analysis

### Expected Results (Reference)
Based on Intel N100 specifications and thermal design:

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Throughput | 15-25 FPS | Depends on thermal conditions and turbo boost |
| P99 Latency | 50-80 ms | Lower with sustained turbo frequencies |
| Mean Latency | 40-67 ms | Should be stable across runs |
| Power Consumption | 6-15 W | Total system power including peripherals |
| CPU Temperature | 45-75°C | With adequate cooling |

## ⚙️ Configuration Options

### Benchmark Parameters
You can modify these parameters in the script:

```bash
NUM_RUNS=5              # Number of benchmark iterations
CPU_AFFINITY="0-3"      # CPU cores to use (all 4 cores)
TIMEOUT=300             # Maximum runtime per iteration (seconds)
```

### Platform Optimizations
The benchmark automatically applies Intel N100-specific optimizations:

- **CPU Governor**: Set to 'performance' mode for maximum sustained performance
- **CPU Frequencies**: Locked to maximum values where possible
- **Turbo Boost**: Enabled by default (can be disabled for consistency)
- **CPU Affinity**: Bound to all 4 CPU cores
- **Intel GPU**: Set to maximum frequency (for monitoring, not used in benchmark)

## 🔧 Troubleshooting

### Common Issues

#### 1. ORB-SLAM3 Not Found
```bash
# Error: ORB-SLAM3 not found
# Solution: Run platform setup first
cd ../setup/
./install_all.sh
```

#### 2. Dataset Missing
```bash
# Error: EuRoC dataset not found
# Solution: Download dataset
cd ../../datasets/
./prepare_all_datasets.sh
```

#### 3. Build Failures
```bash
# If ORB-SLAM3 build fails, try with more conservative settings
cd ~/ORB_SLAM3/
# Edit CMakeLists.txt to use fewer parallel jobs if needed
make -j2  # Instead of -j$(nproc)
```

#### 4. Performance Issues

**Thermal Throttling**:
- Check CPU temperatures: `sensors | grep Core`
- Ensure heatsink is properly mounted with thermal paste
- Verify fan operation: `sudo pwm-config` (if available)
- Monitor frequencies: `watch -n1 "cat /proc/cpuinfo | grep MHz"`

**Low Performance**:
- Verify CPU governor: `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
- Check current frequencies: `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq`
- Ensure adequate power supply: USB-C PD 45W minimum
- Check for background processes: `htop`

**Inconsistent Results**:
- Disable turbo boost for consistency: `echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo`
- Close unnecessary applications
- Check system load: `uptime`

#### 5. Memory Issues
```bash
# Check available memory
free -h

# If memory is limited, reduce ORB-SLAM3 parameters
# Edit EuRoC.yaml to reduce vocabulary size or feature counts
```

### Debug Mode
Run with debug output for troubleshooting:
```bash
bash -x run_benchmark.sh
```

### Manual Performance Tuning
```bash
# Set specific CPU frequency (example: 2.8 GHz)
sudo cpupower frequency-set -f 2800MHz

# Check Intel P-state driver status
cat /sys/devices/system/cpu/intel_pstate/status

# Monitor real-time performance
sudo turbostat --interval 1
```

## 🔬 Scientific Methodology

### Experimental Controls
- **Thermal Stability**: Active cooling prevents thermal throttling
- **Process Isolation**: CPU affinity ensures consistent resource allocation
- **Statistical Significance**: Multiple runs (default: 5) with statistical analysis
- **Quiescent State**: System idle verification before each run
- **Frequency Monitoring**: Continuous CPU/GPU frequency logging

### Measurement Accuracy
- **Hardware-based Power**: External power analyzer for ground truth
- **High-resolution Timing**: Microsecond precision timing measurements
- **System Monitoring**: Continuous CPU, GPU, and thermal monitoring
- **Reproducibility**: Automated setup and consistent environment

### Intel-Specific Considerations
- **Turbo Boost Behavior**: Intel N100 turbo boost affects performance variability
- **Thermal Design Power**: 6W TDP may limit sustained performance
- **Memory Architecture**: Single-channel LPDDR5 may limit memory-intensive workloads
- **x86 vs ARM**: Different instruction set architectures affect performance characteristics

## 📖 Advanced Usage

### Custom CPU Affinity
```bash
# Run on specific CPU cores only
taskset -c 0,1 ./run_benchmark.sh  # Use only first 2 cores
```

### Turbo Boost Control
```bash
# Disable turbo boost for consistent results
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Re-enable turbo boost
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

### Extended Analysis
```bash
# Analyze frequency scaling behavior
cd ~/benchmark_workspace/results/orb_slam3/
python3 -c "
import matplotlib.pyplot as plt
import pandas as pd
# Load frequency monitoring data
# Create frequency vs performance correlation plots
"
```

### Power Measurement Integration
```bash
# For USB-C PD power measurement (requires special equipment)
# Or use software-based power estimation
sudo powertop --csv=power_data.csv --time=300s &
POWERTOP_PID=$!

# Run benchmark
./run_benchmark.sh

# Stop power monitoring
kill $POWERTOP_PID
```

### Comparison with ARM Platforms
This benchmark enables direct comparison with ARM-based platforms:
- **Architecture**: x86 vs ARM instruction sets
- **Manufacturing Process**: Intel 10nm vs ARM 7nm/5nm
- **Memory**: LPDDR5 vs DDR4/5 performance
- **Power Efficiency**: Performance per watt analysis

## 📚 References

1. [ORB-SLAM3 Original Paper](https://arxiv.org/abs/2007.11898)
2. [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
3. [Intel N100 Processor Specifications](https://ark.intel.com/content/www/us/en/ark/products/231803/intel-processor-n100-6m-cache-up-to-3-40-ghz.html)
4. [Radxa X4 Technical Documentation](https://docs.radxa.com/en/x/x4)
5. [Scientific Benchmarking Methodology](../../docs/methodology.md)

## 🤝 Contributing

When modifying this benchmark:
1. Test on actual Radxa X4 hardware with Intel N100
2. Document thermal conditions and power supply specifications
3. Validate against other x86 embedded platforms
4. Maintain compatibility with Ubuntu 20.04 LTS
5. Follow established scientific methodology

---

**Note**: Intel N100 performance is highly dependent on thermal design and power delivery. Always report cooling solution, ambient temperature, and power supply specifications with results. The 6W TDP design may result in thermal throttling under sustained loads without adequate cooling.
