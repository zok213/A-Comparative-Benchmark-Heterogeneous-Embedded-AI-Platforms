# ORB-SLAM3 Benchmark - NVIDIA Jetson

This directory contains the complete benchmarking setup for ORB-SLAM3 Visual-Inertial SLAM on NVIDIA Jetson platforms.

## Overview

ORB-SLAM3 is a feature-based SLAM system that performs real-time tracking, mapping, and loop closure detection. This benchmark evaluates the CPU performance characteristics of ORB-SLAM3 on NVIDIA Jetson platforms using the EuRoC MAV dataset.

### Benchmark Objectives

- **Primary Metric**: Processing throughput (FPS)
- **Secondary Metrics**: CPU utilization, memory usage, power consumption
- **Accuracy Metric**: Absolute Trajectory Error (ATE) in meters
- **Platform Focus**: Multi-core ARM Cortex-A78AE CPU performance

## Prerequisites

### Hardware Requirements
- NVIDIA Jetson Orin NX (8GB or 16GB)
- Active cooling solution (fan or heatsink)
- MicroSD card (64GB+, Class 10 or better)
- Power measurement equipment (optional but recommended)

### Software Requirements
- Ubuntu 20.04 LTS (JetPack 5.1.1)
- OpenCV 4.5+ with CUDA support
- Eigen3 library
- Pangolin for visualization
- ORB-SLAM3 source code and dependencies

### Dataset Requirements
- EuRoC MAV dataset (Machine Hall sequences recommended)
- Approximately 2GB storage space

## Quick Start

1. **Ensure prerequisites are installed:**
   ```bash
   cd ../setup
   ./install_all.sh
   ```

2. **Run the benchmark:**
   ```bash
   ./run_benchmark.sh
   ```

3. **View results:**
   ```bash
   ls results/
   # Check the latest timestamped results folder
   ```

## Benchmark Configuration

### Default Settings
- **Sequences**: MH_01_easy, MH_02_easy, MH_03_medium
- **Runs per sequence**: 3 (for statistical reliability)
- **Warm-up**: 30-second system stabilization
- **Monitoring**: CPU frequency, temperature, memory usage

### Performance Optimizations
The benchmark script automatically applies:
- CPU governor set to 'performance'
- Maximum CPU frequencies enabled
- Memory frequency optimization
- Process priority adjustments
- Thermal throttling monitoring

## Understanding Results

### Output Files
- `benchmark_results.json`: Complete benchmark data
- `system_info.json`: Hardware and software configuration
- `performance_summary.txt`: Human-readable summary
- `trajectory_*.txt`: SLAM trajectory outputs
- `timing_*.log`: Detailed timing information

### Key Metrics
- **Average FPS**: Mean processing rate across all sequences
- **P99 Latency**: 99th percentile frame processing time
- **CPU Utilization**: Average CPU usage during processing
- **Memory Peak**: Maximum memory consumption
- **ATE RMSE**: Trajectory accuracy (lower is better)

### Expected Performance Ranges

#### Jetson Orin NX 16GB
- **Throughput**: 15-25 FPS (depending on sequence complexity)
- **CPU Usage**: 60-80% (utilizing multiple cores)
- **Memory**: 1.5-2.5 GB peak usage
- **Power**: 15-25W during processing

#### Jetson Orin NX 8GB
- **Throughput**: 12-20 FPS
- **CPU Usage**: 70-85%
- **Memory**: 1.5-2.0 GB peak usage
- **Power**: 12-20W during processing

## Configuration Options

### Environment Variables
```bash
# Customize benchmark behavior
export ORBSLAM3_NUM_RUNS=5          # Number of runs per sequence
export ORBSLAM3_SEQUENCES="MH_01_easy,MH_04_difficult"  # Custom sequences
export ORBSLAM3_VOCAB_PATH="/path/to/vocab"             # Custom vocabulary
export ORBSLAM3_ENABLE_PANGOLIN=0   # Disable visualization for headless
```

### Sequence Selection
Edit the `SEQUENCES` array in `run_benchmark.sh`:
```bash
SEQUENCES=(
    "MH_01_easy"
    "MH_02_easy"
    "MH_03_medium"
    "MH_04_difficult"  # Add more challenging sequences
)
```

## Troubleshooting

### Common Issues

#### Low Performance
- **Check thermal throttling**: Monitor temperatures in results
- **Verify cooling**: Ensure active cooling is working
- **Check power mode**: Confirm maximum performance mode
- **Memory constraints**: Monitor for memory pressure

#### Build Errors
```bash
# Rebuild ORB-SLAM3 dependencies
cd ../setup
./install_all.sh --rebuild-orbslam3
```

#### Dataset Issues
```bash
# Re-download EuRoC dataset
cd ../../../datasets
./prepare_all_datasets.sh --euroc-only
```

### Performance Debugging
```bash
# Run with detailed profiling
ORBSLAM3_PROFILE=1 ./run_benchmark.sh

# Monitor system during benchmark
htop  # In another terminal
```

## Scientific Methodology

### Measurement Protocol
1. **System Preparation**: 5-minute idle period for thermal stabilization
2. **Warm-up**: 30-second processing of dummy data
3. **Measurement**: Multiple runs with statistical analysis
4. **Validation**: Trajectory accuracy verification

### Statistical Analysis
- Mean and standard deviation across runs
- Outlier detection and removal
- Confidence intervals (95%)
- Performance consistency metrics

### Reproducibility
- Fixed random seeds where applicable
- Consistent system configuration
- Detailed environment logging
- Version-controlled parameters

## Advanced Usage

### Custom Vocabulary
```bash
# Use custom ORB vocabulary
export ORBSLAM3_VOCAB_PATH="/path/to/custom/vocab.txt"
./run_benchmark.sh
```

### Multi-Session Analysis
```bash
# Compare multiple benchmark runs
python3 ../../../analysis/scripts/compare_orbslam_results.py \
    results/run_2024-01-15_10-30-00/ \
    results/run_2024-01-15_14-20-00/
```

### Power Profiling Integration
```bash
# Enable power measurement (requires Yokogawa WT300E)
export ENABLE_POWER_MEASUREMENT=1
export POWER_METER_IP="192.168.1.100"
./run_benchmark.sh
```

## References

- [ORB-SLAM3 Paper](https://arxiv.org/abs/2007.11898)
- [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
- [NVIDIA Jetson Performance Guide](https://docs.nvidia.com/jetson/)

## Support

For issues specific to this benchmark:
1. Check the troubleshooting section above
2. Review system requirements in `../setup/README.md`
3. Consult the main project documentation in `../../../docs/`
