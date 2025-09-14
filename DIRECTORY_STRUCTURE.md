# Complete Directory Structure

This document provides a comprehensive overview of the embedded AI benchmark suite directory structure.

## 📁 Root Directory Structure

```
embedded-ai-benchmark-suite/
├── 📄 README.md                           # Main project overview and quick start
├── 📄 USAGE_GUIDE.md                      # Comprehensive usage instructions
├── 📄 DIRECTORY_STRUCTURE.md              # This file - complete directory guide
├── 📄 run_all_benchmarks.sh               # Master orchestration script
├── 📁 platforms/                          # Platform-specific implementations
├── 📁 datasets/                           # Dataset preparation and management
├── 📁 analysis/                           # Results analysis and visualization
├── 📁 docs/                               # Technical documentation
└── 📁 reference-guides/                   # Original methodology papers
```

## 🏗️ Platforms Directory (`platforms/`)

Contains complete implementations for each target embedded AI platform:

```
platforms/
├── 📁 nvidia-jetson/                      # NVIDIA Jetson Orin NX platform
│   ├── 📄 README.md                       # Complete platform setup guide
│   ├── 📁 setup/                          # Installation and configuration
│   │   ├── 📄 install_all.sh             # Automated setup script
│   │   └── 📄 README.md                  # Setup instructions
│   ├── 📁 orb-slam3/                      # ORB-SLAM3 CPU benchmark
│   │   ├── 📄 run_benchmark.sh           # Benchmark execution script
│   │   └── 📄 README.md                  # Detailed benchmark guide
│   ├── 📁 3d-object-detection/            # 3D Object Detection pipeline
│   │   ├── 📄 run_benchmark.sh           # Benchmark execution script
│   │   └── 📄 README.md                  # Detailed benchmark guide
│   └── 📁 semantic-segmentation/          # Semantic Segmentation benchmark
│       ├── 📄 run_benchmark.sh           # Benchmark execution script
│       └── 📄 README.md                  # Detailed benchmark guide
├── 📁 qualcomm-qcs6490/                   # Qualcomm QCS6490 platform
│   ├── 📄 README.md                       # Complete platform setup guide
│   ├── 📁 setup/                          # SNPE SDK installation & config
│   │   ├── 📄 install_all.sh             # Automated setup script
│   │   └── 📄 README.md                  # Setup instructions
│   ├── 📁 orb-slam3/                      # ORB-SLAM3 CPU benchmark
│   │   ├── 📄 run_benchmark.sh           # Benchmark execution script
│   │   └── 📄 README.md                  # Detailed benchmark guide
│   ├── 📁 3d-object-detection/            # 3D Object Detection pipeline
│   │   ├── 📄 run_benchmark.sh           # Benchmark execution script
│   │   └── 📄 README.md                  # Detailed benchmark guide
│   └── 📁 semantic-segmentation/          # Semantic Segmentation benchmark
│       ├── 📄 run_benchmark.sh           # Benchmark execution script
│       └── 📄 README.md                  # Detailed benchmark guide
└── 📁 radxa-x4/                           # Radxa X4 (Intel N100) platform
    ├── 📄 README.md                       # Complete platform setup guide
    ├── 📁 setup/                          # OpenVINO installation & config
    │   ├── 📄 install_all.sh             # Automated setup script
    │   └── 📄 README.md                  # Setup instructions
    ├── 📁 orb-slam3/                      # ORB-SLAM3 CPU benchmark
    │   ├── 📄 run_benchmark.sh           # Benchmark execution script
    │   └── 📄 README.md                  # Detailed benchmark guide
    ├── 📁 3d-detection/                   # 3D Object Detection pipeline
    │   ├── 📄 run_benchmark.sh           # Benchmark execution script
    │   └── 📄 README.md                  # Detailed benchmark guide
    └── 📁 segmentation/                   # Semantic Segmentation benchmark
        ├── 📄 run_benchmark.sh           # Benchmark execution script
        └── 📄 README.md                  # Detailed benchmark guide
```

## 📊 Datasets Directory (`datasets/`)

Centralized dataset preparation and management:

```
datasets/
├── 📄 prepare_all_datasets.sh             # Master dataset preparation script
├── 📄 README.md                           # Dataset setup guide
├── 📁 euroc/                              # EuRoC MAV dataset (ORB-SLAM3)
│   └── 📁 MH01/                           # Machine Hall 01 sequence
├── 📁 kitti/                              # KITTI dataset (3D detection)
│   └── 📁 object/
│       ├── 📁 training/
│       │   ├── 📁 image_2/                # Left stereo images
│       │   ├── 📁 image_3/                # Right stereo images
│       │   ├── 📁 calib/                  # Camera calibration
│       │   └── 📁 label_2/                # 3D object labels
│       └── 📁 testing/
└── 📁 cityscapes/                         # Cityscapes dataset (segmentation)
    ├── 📁 leftImg8bit/
    │   ├── 📁 train/
    │   ├── 📁 val/
    │   └── 📁 test/
    └── 📁 gtFine/
        ├── 📁 train/
        ├── 📁 val/
        └── 📁 test/
```

## 📈 Analysis Directory (`analysis/`)

Results processing and visualization tools:

```
analysis/
├── 📁 scripts/                            # Data processing scripts
│   ├── 📄 analyze_all_results.py         # Comprehensive analysis script
│   ├── 📄 power_analysis.py              # Power measurement analysis
│   ├── 📄 performance_comparison.py      # Cross-platform comparison
│   └── 📄 README.md                      # Analysis guide
├── 📁 notebooks/                          # Jupyter analysis notebooks
│   ├── 📄 platform_comparison.ipynb      # Interactive comparison analysis
│   ├── 📄 power_efficiency.ipynb         # Power efficiency analysis
│   └── 📄 accuracy_validation.ipynb      # Accuracy metric validation
└── 📁 templates/                          # Report templates
    ├── 📄 benchmark_report_template.md   # Standard report format
    └── 📄 comparison_table_template.md   # Cross-platform comparison format
```

## 📚 Documentation Directory (`docs/`)

Technical documentation and guides:

```
docs/
├── 📄 hardware-requirements.md            # Detailed hardware specifications
├── 📄 power-measurement-setup.md          # Power measurement guide
├── 📄 methodology.md                      # Scientific benchmarking methodology
├── 📄 troubleshooting.md                  # Common issues and solutions
├── 📄 software-requirements.md            # Software dependencies
├── 📄 calibration-procedures.md           # Model calibration procedures
└── 📁 diagrams/                           # Technical diagrams and schematics
    ├── 📄 power-measurement-wiring.png    # Power analyzer wiring diagram
    ├── 📄 platform-architecture.png       # Platform architecture comparison
    └── 📄 benchmark-pipeline.png          # Benchmark execution flow
```

## 📖 Reference Guides Directory (`reference-guides/`)

Comprehensive theoretical and implementation guides:

```
reference-guides/
├── 📄 README.md                            # Overview of reference documentation
├── 📄 3D-Object-Detection-Reference-Guide.md # Complete 3D detection pipeline guide
├── 📄 ORB-SLAM3-Reference-Guide.md         # Visual-inertial SLAM reference
└── 📄 Semantic-Segmentation-Reference-Guide.md # Real-time segmentation guide
```

## 🏠 Runtime Directory Structure

When benchmarks are executed, the following directory structure is created in the user's home directory:

```
~/benchmark_workspace/
├── 📄 setup_env.sh                        # Environment configuration script
├── 📁 datasets/                           # Symlink to datasets directory
├── 📁 models/                             # Platform-specific optimized models
│   ├── 📁 onnx/                           # Original ONNX models
│   ├── 📁 tensorrt/                       # TensorRT engines (NVIDIA)
│   ├── 📁 snpe/                           # SNPE DLC files (Qualcomm)
│   └── 📁 openvino/                       # OpenVINO IR files (Intel)
├── 📁 results/                            # Benchmark results
│   ├── 📁 orb_slam3/                      # ORB-SLAM3 results
│   ├── 📁 3d_detection/                   # 3D detection results
│   └── 📁 segmentation/                   # Segmentation results
├── 📁 logs/                               # System and benchmark logs
└── 📁 scripts/                            # Generated helper scripts
```

## 🔄 Workflow Directory Navigation

### For Platform Setup:
```bash
cd platforms/nvidia-jetson/setup/          # NVIDIA Jetson setup
cd platforms/qualcomm-qcs6490/setup/       # Qualcomm QCS6490 setup
cd platforms/radxa-x4/setup/               # Radxa X4 setup
```

### For Running Benchmarks:
```bash
cd platforms/PLATFORM/BENCHMARK/           # Individual benchmark
# Examples:
cd platforms/nvidia-jetson/orb-slam3/      # NVIDIA ORB-SLAM3
cd platforms/qualcomm-qcs6490/3d-detection/ # Qualcomm 3D detection
cd platforms/radxa-x4/segmentation/        # Radxa segmentation
```

### For Dataset Preparation:
```bash
cd datasets/                               # Dataset preparation
```

### For Results Analysis:
```bash
cd analysis/scripts/                       # Analysis scripts
cd analysis/notebooks/                     # Jupyter notebooks
```

## 📋 File Naming Conventions

### Scripts:
- `install_all.sh` - Complete platform setup
- `run_benchmark.sh` - Individual benchmark execution
- `prepare_all_datasets.sh` - Dataset preparation
- `analyze_all_results.py` - Results analysis

### Documentation:
- `README.md` - Primary documentation for each directory
- `*-methodology.md` - Detailed technical methodology
- `*-requirements.md` - Requirements and specifications
- `*-setup.md` - Setup and installation guides

### Results:
- `*_results_*.json` - Structured benchmark results
- `*_analysis.txt` - Human-readable analysis
- `*_visualization.png` - Performance visualizations
- `*_comparison.csv` - Cross-platform comparisons

## 🎯 Directory Purpose Summary

| Directory | Primary Purpose | Key Files |
|-----------|----------------|-----------|
| `platforms/` | Platform-specific implementations | `README.md`, `run_benchmark.sh` |
| `datasets/` | Dataset management | `prepare_all_datasets.sh` |
| `analysis/` | Results processing | `analyze_all_results.py` |
| `docs/` | Technical documentation | `hardware-requirements.md` |
| `reference-guides/` | Methodology papers | `*-methodology.md` |

This structure ensures:
- **Clear separation** of platform-specific code
- **Centralized dataset** management
- **Comprehensive documentation** at every level
- **Easy navigation** for users and contributors
- **Scalable architecture** for adding new platforms or benchmarks
