# ORB-SLAM3 Benchmark - Qualcomm QCS6490

This directory contains the complete setup and execution scripts for benchmarking ORB-SLAM3 performance on the Qualcomm QCS6490 platform.

## 📋 Overview

**Benchmark Type**: CPU Performance Evaluation  
**Target Platform**: Qualcomm QCS6490 (Kryo 670 CPU)  
**Workload**: Visual-Inertial SLAM using ORB-SLAM3  
**Dataset**: EuRoC MAV Machine Hall 01 sequence  
**Key Metrics**: Throughput (FPS), P99 Latency, Power Consumption  

## 🎯 Benchmark Objective

This benchmark evaluates the CPU and memory subsystem performance of the Qualcomm QCS6490 platform using ORB-SLAM3, a computationally intensive visual-inertial SLAM algorithm. ORB-SLAM3 is chosen because it:

- **Multi-threaded workload**: Stresses all CPU cores simultaneously
- **Memory intensive**: Tests memory bandwidth and cache hierarchy
- **Real-world application**: Represents actual robotics/AR workloads
- **CPU-bound**: Does not rely on specialized AI accelerators

## 🛠️ Prerequisites

### Hardware Requirements
- Qualcomm QCS6490 development board (Thundercomm TurboX C6490 recommended)
- Active cooling solution (40mm fan + heatsink) - **MANDATORY**
- Stable 12V/3A power supply
- microSD card (64GB+) or eUFS storage
- Yokogawa WT300E power analyzer (for accurate power measurement)

### Software Requirements
- Ubuntu 20.04 LTS (ARM64)
- Platform setup completed (run `../setup/install_all.sh` first)
- EuRoC MAV dataset downloaded (run `../../datasets/prepare_all_datasets.sh`)

## 🚀 Quick Start

### 1. Verify Prerequisites
```bash
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
cd qualcomm-qcs6490/orb-slam3/
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

### 4. Analyze Results with Unified Analysis Scripts
```bash
# Verify all metrics are correct and complete
python ../../analysis/orb_slam3_verify_metrics.py

# Calculate comprehensive performance metrics
python ../../analysis/orb_slam3_calculate_metrics.py

# Generate advanced analysis for research/publication
python ../../analysis/orb_slam3_advanced_analysis.py
```

## 📊 Understanding Results

### Output Files
```
~/benchmark_workspace/results/orb_slam3/
├── summary.txt                    # Basic run summary
├── detailed_analysis.txt          # Comprehensive performance metrics
├── performance_analysis.png       # Performance visualization
├── logs/                          # Individual run logs
│   ├── run_1_YYYYMMDD_HHMMSS.log
│   ├── run_2_YYYYMMDD_HHMMSS.log
│   └── ...
├── system_monitor_*.log           # System monitoring data
└── cpu_freq_*.log                 # CPU frequency monitoring
```

### Key Metrics Explained

#### Performance Metrics
- **Throughput (FPS)**: Average frames processed per second
  - Higher values indicate better performance
  - Typical range: 5-25 FPS depending on CPU performance
  
- **P99 Latency (ms)**: 99th percentile frame processing time
  - Represents worst-case performance for 99% of frames
  - Critical for real-time applications
  - Lower values are better

- **Mean Latency (ms)**: Average frame processing time
  - Overall processing speed indicator
  - Should be consistent across runs

#### System Metrics
- **CPU Utilization**: Multi-core CPU usage during benchmark
- **Memory Usage**: RAM consumption patterns
- **Thermal Behavior**: CPU temperature and throttling events

### Expected Results (Reference)
Based on the Qualcomm Kryo 670 CPU architecture:

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Throughput | 8-15 FPS | Depends on thermal conditions |
| P99 Latency | 80-150 ms | Varies with CPU frequency |
| Mean Latency | 65-125 ms | Should be stable across runs |
| Power Consumption | 8-15 W | Total system power |

## ⚙️ Configuration Options

### Benchmark Parameters
You can modify these parameters in the script:

```bash
NUM_RUNS=5              # Number of benchmark iterations
CPU_AFFINITY="0-3"      # CPU cores to use (0-3 for efficiency cores)
TIMEOUT=300             # Maximum runtime per iteration (seconds)
```

### Platform Optimizations
The benchmark automatically applies QCS6490-specific optimizations:

- **CPU Governor**: Set to 'performance' mode
- **CPU Frequencies**: Locked to maximum values
- **CPU Affinity**: Bound to specific cores for consistency
- **CPU Hotplug**: Disabled to prevent core switching

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

#### 3. Build Failures (Out of Memory)
```bash
# If ORB-SLAM3 build fails due to memory issues
cd ~/ORB_SLAM3/
# Edit build.sh to use fewer parallel jobs
sed -i 's/make -j/make -j2/g' build.sh
./build.sh
```

#### 4. Performance Issues

**Thermal Throttling**:
- Ensure active cooling is working
- Check CPU temperatures: `cat /sys/class/thermal/thermal_zone*/temp`
- Verify fan operation and heatsink contact

**Inconsistent Results**:
- Close all unnecessary applications
- Ensure system is idle before benchmark
- Check for background processes: `htop`

**Low Performance**:
- Verify CPU governor: `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
- Check CPU frequencies: `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq`
- Ensure power supply is adequate (12V/3A minimum)

### Debug Mode
Run with debug output for troubleshooting:
```bash
bash -x run_benchmark.sh
```

### Log Analysis
For detailed debugging, examine the individual run logs:
```bash
# View latest run log
ls -la ~/benchmark_workspace/results/orb_slam3/logs/
tail -100 ~/benchmark_workspace/results/orb_slam3/logs/run_1_*.log
```

## 🔬 Unified Analysis Framework

This benchmark leverages the project's **unified analysis system** for consistent, scientific-grade evaluation across all embedded AI platforms:

### 📊 Analysis Tools Overview

**`orb_slam3_verify_metrics.py`** - Comprehensive Verification
- ✅ Validates trajectory accuracy calculations (RMSE, ATE, RPE) 
- 🔍 Checks dataset integrity and ground truth synchronization
- 📋 Ensures benchmark compliance with TUM RGB-D standards
- 🎯 Verifies statistical correctness and reproducibility

**`orb_slam3_calculate_metrics.py`** - Performance Analysis  
- ⚡ Calculates platform-specific performance characteristics
- 💾 Analyzes memory utilization and bottlenecks
- 🔋 Evaluates power efficiency metrics (fps/W)
- 🏆 Generates standardized benchmark scores (0-100 scale)

**`orb_slam3_advanced_analysis.py`** - Research-Grade Analysis
- 📈 Advanced statistical analysis with confidence intervals
- 🔄 Cross-platform comparative analysis vs other embedded AI platforms
- 📚 Publication-ready insights and research contributions
- 🔬 Peer-review quality scientific rigor assessment

### 🎯 Platform-Specific Insights for QCS6490

The unified scripts automatically detect Qualcomm QCS6490 and provide:
- **Power Efficiency Focus**: Industry-leading fps/W analysis
- **Mobile Optimization**: DSP integration potential assessment  
- **Thermal Analysis**: Kryo CPU thermal behavior characterization
- **Adreno GPU**: Graphics subsystem utilization analysis

### 💡 Benefits of Unified Approach
- **Cross-Platform Comparison**: Direct comparison with NVIDIA Jetson, Radxa CM5
- **Consistency**: Same methodology ensures fair benchmarking
- **Research Quality**: Publication-ready statistical analysis
- **Maintenance**: Single source of truth, automatic updates

### 📋 Analysis Workflow Example
```bash
# After benchmark completion
./run_benchmark.sh

# Comprehensive analysis pipeline  
python ../../analysis/orb_slam3_verify_metrics.py      # ✅ Verification
python ../../analysis/orb_slam3_calculate_metrics.py   # 📊 Performance  
python ../../analysis/orb_slam3_advanced_analysis.py   # 🔬 Research analysis
```

## 🔬 Scientific Methodology

### Experimental Controls
- **Thermal Stability**: Active cooling prevents thermal throttling
- **Process Isolation**: CPU affinity ensures consistent resource allocation
- **Statistical Significance**: Multiple runs (default: 5) with statistical analysis
- **Quiescent State**: System idle verification before each run

### Measurement Accuracy
- **Hardware-based Power**: External power analyzer for ground truth
- **High-resolution Timing**: Microsecond precision timing measurements
- **System Monitoring**: Continuous CPU and thermal monitoring

### Reproducibility
- **Automated Setup**: Complete environment configuration
- **Version Control**: Specific software versions documented
- **Configuration Logging**: All settings recorded in results

## 📖 Advanced Usage

### Custom CPU Affinity
```bash
# Run on specific CPU cores (e.g., performance cores only)
taskset -c 4-7 ./run_benchmark.sh
```

### Extended Analysis
```bash
# Generate additional analysis plots
cd ~/benchmark_workspace/results/orb_slam3/
python3 -c "
import matplotlib.pyplot as plt
import numpy as np
# Add custom analysis code here
"
```

### Power Measurement Integration
If using Yokogawa WT300E power analyzer:
```bash
# Start power logging before benchmark
python3 ../../docs/power_logger.py --output power_data.csv &
POWER_PID=$!

# Run benchmark
./run_benchmark.sh

# Stop power logging
kill $POWER_PID

# Analyze power data
python3 ../../docs/analyze_power.py power_data.csv
```

## 📚 References

1. [ORB-SLAM3 Original Paper](https://arxiv.org/abs/2007.11898)
2. [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
3. [Qualcomm QCS6490 Technical Specifications](https://www.qualcomm.com/products/qcs6490)
4. [Scientific Benchmarking Methodology](../../docs/methodology.md)

## 🤝 Contributing

When modifying this benchmark:
1. Maintain scientific rigor and reproducibility
2. Document all changes in commit messages
3. Test on actual QCS6490 hardware
4. Update expected results if hardware changes
5. Follow the established coding style

---

**Note**: This benchmark is designed for research and evaluation purposes. Results should be interpreted in the context of the specific hardware configuration, thermal conditions, and software versions used.
