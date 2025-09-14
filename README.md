# Embedded AI Benchmark Suite

A comprehensive, scientifically rigorous benchmarking framework for evaluating heterogeneous embedded AI platforms across Visual-Inertial SLAM, 3D Object Detection, and Semantic Segmentation workloads.

## 🎯 Overview

This repository provides complete setup guides, scripts, and analysis tools for benchmarking three distinct embedded AI platforms with three different computational workloads:

### Target Platforms
- **NVIDIA Jetson Orin NX** - GPU-centric architecture with TensorRT optimization
- **Qualcomm QCS6490** - DSP/NPU-centric architecture with SNPE optimization  
- **Radxa X4 (Intel N100)** - x86-based with integrated Intel UHD Graphics and OpenVINO optimization

### Benchmark Workloads
- **ORB-SLAM3** - Visual-Inertial SLAM (CPU-bound, multi-threaded)
- **3D Object Detection** - Pseudo-LiDAR + PointPillars pipeline (AI accelerator workload)
- **Semantic Segmentation** - DDRNet-23-slim (lightweight real-time AI workload)

## 🏗️ Project Structure

```
📦 embedded-ai-benchmark-suite/
├── 📁 platforms/                     # Platform-specific implementations
│   ├── 📁 nvidia-jetson/            # NVIDIA Jetson Orin NX
│   │   ├── 📄 README.md             # Complete platform setup guide
│   │   ├── 📁 setup/                # Installation and configuration
│   │   │   ├── 📄 install_all.sh    # Automated setup script
│   │   │   └── 📄 README.md         # Setup instructions
│   │   ├── 📁 orb-slam3/            # ORB-SLAM3 CPU benchmark
│   │   │   ├── 📄 run_benchmark.sh  # Benchmark execution script
│   │   │   └── 📄 README.md         # Detailed benchmark guide
│   │   ├── 📁 3d-object-detection/  # 3D Object Detection pipeline
│   │   │   ├── 📄 run_benchmark.sh  # Benchmark execution script
│   │   │   └── 📄 README.md         # Detailed benchmark guide
│   │   └── 📁 semantic-segmentation/ # Semantic Segmentation
│   │       ├── 📄 run_benchmark.sh  # Benchmark execution script
│   │       └── 📄 README.md         # Detailed benchmark guide
│   ├── 📁 qualcomm-qcs6490/         # Qualcomm QCS6490 Platform
│   │   ├── 📄 README.md             # Complete platform setup guide
│   │   ├── 📁 setup/                # SNPE SDK installation & config
│   │   ├── 📁 orb-slam3/            # ORB-SLAM3 CPU benchmark
│   │   ├── 📁 3d-object-detection/  # 3D Object Detection pipeline
│   │   └── 📁 semantic-segmentation/ # Semantic Segmentation
│   └── 📁 radxa-x4/                 # Radxa X4 (Intel N100)
│       ├── 📄 README.md             # Complete platform setup guide
│       ├── 📁 setup/                # OpenVINO installation & config
│       ├── 📁 orb-slam3/            # ORB-SLAM3 CPU benchmark
│       ├── 📁 3d-detection/         # 3D Object Detection pipeline
│       └── 📁 segmentation/         # Semantic Segmentation
├── 📁 datasets/                      # Dataset preparation and management
│   ├── 📄 prepare_all_datasets.sh   # Master dataset preparation script
│   ├── 📄 README.md                 # Dataset setup guide
│   ├── 📁 euroc/                    # EuRoC MAV dataset for ORB-SLAM3
│   ├── 📁 kitti/                    # KITTI dataset for 3D detection
│   └── 📁 cityscapes/               # Cityscapes dataset for segmentation
├── 📁 analysis/                      # Results analysis and visualization
│   ├── 📁 scripts/                  # Data processing and analysis scripts
│   │   ├── 📄 analyze_all_results.py # Comprehensive analysis script
│   │   └── 📄 README.md             # Analysis guide
│   └── 📁 notebooks/                # Jupyter analysis notebooks
├── 📁 docs/                          # Comprehensive documentation
│   ├── 📄 hardware-requirements.md  # Detailed hardware specifications
│   ├── 📄 power-measurement-setup.md # Power measurement guide
│   ├── 📄 methodology.md            # Scientific benchmarking methodology
│   └── 📄 troubleshooting.md        # Common issues and solutions
├── 📁 reference-guides/              # Original benchmark methodology papers
│   ├── 📄 orb-slam3-methodology.md  # ORB-SLAM3 benchmarking guide
│   ├── 📄 3d-detection-methodology.md # 3D detection pipeline guide
│   └── 📄 segmentation-methodology.md # Semantic segmentation guide
├── 📄 run_all_benchmarks.sh         # Master orchestration script
├── 📄 USAGE_GUIDE.md                # Comprehensive usage instructions
└── 📄 README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites
- Target hardware platform (Jetson Orin NX, QCS6490, or Radxa X4)
- Ubuntu 20.04 LTS installed on target platform
- Active cooling solution (mandatory for thermal stability)
- Yokogawa WT300E power analyzer (recommended for power measurements)
- Host PC for dataset preparation and analysis

### 1. Clone Repository
```bash
git clone <repository-url>
cd embedded-ai-benchmark-suite
```

### 2. Platform Setup
Choose your target platform and follow the setup guide:

```bash
# For NVIDIA Jetson Orin NX
cd platforms/nvidia-jetson/
cat README.md  # Read complete setup guide

# For Qualcomm QCS6490
cd platforms/qualcomm-qcs6490/
cat README.md  # Read complete setup guide

# For Radxa X4 (Intel N100)
cd platforms/radxa-x4/
cat README.md  # Read complete setup guide
```

### 3. Run Automated Setup
```bash
cd setup/
./install_all.sh
```

### 4. Prepare Datasets
```bash
cd ../../datasets/
./prepare_all_datasets.sh
```

### 5. Run Benchmarks
```bash
# Run individual benchmarks
cd ../platforms/YOUR_PLATFORM/orb-slam3/
./run_benchmark.sh

cd ../3d-detection/
./run_benchmark.sh

cd ../segmentation/
./run_benchmark.sh
```

### 6. Analyze Results
```bash
cd ../../analysis/scripts/
python3 analyze_all_results.py --results-root ../../platforms/ --output-dir ../results/
```

## 📊 Benchmark Results

Each benchmark produces standardized metrics for cross-platform comparison:

### Performance Metrics
- **Throughput (FPS)**: Frames processed per second
- **P99 Latency (ms)**: 99th percentile processing latency
- **Mean Latency (ms)**: Average processing time per frame

### Efficiency Metrics  
- **Average Power (W)**: Power consumption during benchmark execution
- **Power Efficiency (FPS/W)**: Performance per watt consumed

### Accuracy Metrics
- **ORB-SLAM3**: Absolute Trajectory Error (ATE), Relative Pose Error (RPE)
- **3D Detection**: 3D mean Average Precision (mAP) on KITTI
- **Segmentation**: mean Intersection over Union (mIoU) on Cityscapes

## 🔬 Scientific Methodology

This benchmark suite follows rigorous scientific principles:

- **Consistency**: Identical software stacks and datasets across platforms
- **Workload Isolation**: CPU affinity and process isolation for reliable measurements
- **Thermal Stability**: Active cooling prevents thermal throttling
- **Hardware-based Power Measurement**: External power analyzers for ground truth
- **Statistical Significance**: Multiple runs with proper statistical analysis
- **Reproducibility**: Complete documentation and automated setup scripts

## 📖 Documentation

### Platform-Specific Guides
- [NVIDIA Jetson Setup](platforms/nvidia-jetson/README.md)
- [Qualcomm QCS6490 Setup](platforms/qualcomm-qcs6490/README.md)  
- [Radxa X4 Setup](platforms/radxa-x4/README.md)

### Benchmark Guides
Each benchmark has detailed setup and execution guides in its respective directory.

### Technical Documentation
- [Hardware Requirements](docs/hardware-requirements.md)
- [Power Measurement Setup](docs/power-measurement-setup.md)
- [Scientific Methodology](docs/methodology.md)
- [Troubleshooting Guide](docs/troubleshooting.md)

## 🤝 Contributing

Please read our contribution guidelines and ensure all benchmarks follow the established scientific methodology.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this benchmark suite in your research, please cite:

```bibtex
@misc{embedded-ai-benchmark-suite,
  title={Embedded AI Benchmark Suite: A Comprehensive Framework for Evaluating Heterogeneous AI Platforms},
  author={[Your Name]},
  year={2025},
  howpublished={\url{https://github.com/your-repo/embedded-ai-benchmark-suite}}
}
```

## 🆘 Support

For questions, issues, or contributions:
1. Check the [troubleshooting guide](docs/troubleshooting.md)
2. Search existing [GitHub issues](link-to-issues)
3. Create a new issue with detailed information

---

**Note**: This benchmark suite is designed for research and evaluation purposes. Results may vary based on specific hardware configurations, thermal conditions, and software versions.