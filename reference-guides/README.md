# Reference Guides

This directory contains comprehensive reference documentation for each benchmark workload. These guides provide detailed theoretical background, implementation details, and analysis methodologies.

## Contents

### [3D-Object-Detection-Reference-Guide.md](3D-Object-Detection-Reference-Guide.md)
Comprehensive guide covering the 3D Object Detection pipeline using Pseudo-LiDAR + PointPillars:
- **Theoretical Background**: Stereo vision, pseudo-LiDAR generation, 3D object detection
- **Pipeline Architecture**: CREStereo + PointPillars two-stage approach
- **Implementation Details**: Model architectures, optimization techniques
- **Evaluation Methodology**: KITTI dataset, 3D AP metrics, benchmarking protocol
- **Performance Analysis**: Cross-platform comparison, optimization strategies

### [ORB-SLAM3-Reference-Guide.md](ORB-SLAM3-Reference-Guide.md)
Detailed documentation for Visual-Inertial SLAM benchmarking:
- **SLAM Fundamentals**: Visual-inertial odometry, loop closure, mapping
- **ORB-SLAM3 Architecture**: Multi-session SLAM, robust initialization
- **CPU Performance Focus**: Multi-threading, memory optimization
- **Evaluation Protocol**: EuRoC MAV dataset, trajectory accuracy metrics
- **Benchmarking Methodology**: Statistical analysis, reproducibility guidelines

### [Semantic-Segmentation-Reference-Guide.md](Semantic-Segmentation-Reference-Guide.md)
Complete reference for real-time semantic segmentation benchmarking:
- **Segmentation Theory**: Dense prediction, real-time constraints
- **DDRNet Architecture**: Dual-resolution dual-branch design
- **Acceleration Techniques**: TensorRT, SNPE, OpenVINO optimizations
- **Dataset and Metrics**: Cityscapes evaluation, mIoU calculation
- **Performance Optimization**: Quantization, memory management, throughput analysis

## Usage

These reference guides complement the platform-specific implementation guides found in `platforms/*/README.md`. While the platform guides focus on practical setup and execution, these references provide:

1. **Theoretical Foundation**: Understanding the algorithms and their computational characteristics
2. **Implementation Insights**: Design decisions and optimization strategies
3. **Benchmarking Rationale**: Why specific metrics and methodologies were chosen
4. **Cross-Platform Analysis**: Comparative performance insights across different hardware

## Target Audience

- **Researchers**: Seeking detailed understanding of benchmark methodologies
- **Engineers**: Looking for implementation guidance and optimization strategies
- **Students**: Learning about embedded AI and performance evaluation
- **Industry Practitioners**: Evaluating hardware platforms for specific workloads

## Relationship to Platform Guides

```
Reference Guides (this folder)     Platform-Specific Guides
├── Theoretical background    →    ├── Setup instructions
├── Algorithm details        →    ├── Configuration options
├── Benchmarking methodology →    ├── Troubleshooting
└── Cross-platform analysis  →    └── Expected performance
```

## Contributing

When updating these reference guides:
1. Maintain scientific rigor and cite relevant papers
2. Keep implementation details current with the latest platform guides
3. Update performance baselines as new results become available
4. Ensure consistency across all three benchmark domains
