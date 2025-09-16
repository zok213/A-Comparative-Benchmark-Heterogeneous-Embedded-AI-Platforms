# 3D Object Detection Benchmark - NVIDIA J1. **Install dependencies:**
   ```bash
   cd ../setup
   ./install_all.sh
   ```

2. **Download models and datasets:**
   ```bash
   # Option 1: Download both models and datasets together
   python3 ../../../datasets/download_all_datasets.py --datasets kitti --include-models
   
   # Option 2: Download separately with specific control
   python3 ../../../datasets/download_all_models.py --benchmarks 3d-detection
   python3 ../../../datasets/download_kitti_with_cookies.py
   # Downloads: scene_flow + object_calib for stereo depth + 3D detection
   
   # Models downloaded: CREStereo + PointPillars (ONNX format)
   # Platform-specific optimization (TensorRT) happens automatically during benchmark
   ```irectory contains the complete benchmarking setup for 3D Object Detection using Pseudo-LiDAR + PointPillars on NVIDIA Jetson platforms.

## Overview

This benchmark implements a two-stage 3D object detection pipeline:
1. **Stereo Depth Estimation**: CREStereo for dense depth map generation
2. **3D Object Detection**: PointPillars for object detection in pseudo-LiDAR point clouds

The benchmark evaluates both CPU and GPU performance using TensorRT optimization on the KITTI 3D Object Detection dataset.

### Benchmark Objectives

- **Primary Metric**: Detection throughput (FPS)
- **Secondary Metrics**: GPU/CPU utilization, memory usage, power consumption
- **Accuracy Metric**: 3D Average Precision (3D AP) at IoU 0.7
- **Platform Focus**: TensorRT acceleration on Jetson GPU

## Prerequisites

### Hardware Requirements
- NVIDIA Jetson Orin NX (8GB or 16GB)
- Active cooling solution (critical for sustained GPU performance)
- MicroSD card (128GB+, Class 10 or better)
- Power measurement equipment (optional but recommended)

### Software Requirements
- Ubuntu 20.04 LTS (JetPack 5.1.1)
- CUDA 11.4+ with cuDNN
- TensorRT 8.5.2+
- OpenCV 4.5+ with CUDA support
- PyTorch 1.13+ with CUDA support
- ONNX and onnx-tensorrt

### Dataset Requirements
- KITTI 3D Object Detection dataset
- KITTI Stereo 2015 dataset (for depth estimation)
- Approximately 15GB storage space

## Quick Start

1. **Ensure prerequisites are installed:**
   ```bash
   cd ../setup
   ./install_all.sh
   ```

2. **Download KITTI dataset (optimized):**
   ```bash
   # Download only essential files (145MB vs 23GB full dataset)
   python3 ../../../datasets/download_kitti_with_cookies.py
   # Downloads: scene_flow + object_calib for stereo depth + 3D detection
   
   # Alternative: Use the unified dataset downloader
   python3 ../../../datasets/download_all_datasets.py --datasets kitti
   ```

3. **Run the benchmark:**
   ```bash
   ./run_benchmark.sh
   ```

4. **View results:**
   ```bash
   ls results/
   # Check the latest timestamped results folder
   ```

## Benchmark Configuration

### Default Settings
- **Model Variants**: FP32 and INT8 quantized versions
- **Batch Sizes**: 1, 2, 4 (depending on memory constraints)
- **Inference Engines**: TensorRT optimized
- **Test Sequences**: KITTI validation set (200 samples)

### TensorRT Optimizations
The benchmark automatically applies:
- Layer fusion and kernel auto-tuning
- INT8 Post-Training Quantization (PTQ)
- Dynamic shape optimization
- Memory pool optimization
- Multi-stream execution

## Understanding Results

### Output Files
- `benchmark_results.json`: Complete performance metrics
- `system_info.json`: Hardware and software configuration
- `tensorrt_engines/`: Optimized TensorRT engines
- `detection_results/`: 3D bounding box predictions
- `timing_analysis.json`: Detailed latency breakdown

### Key Metrics
- **End-to-End FPS**: Complete pipeline throughput
- **Stereo Depth FPS**: CREStereo processing rate
- **Detection FPS**: PointPillars inference rate
- **3D AP**: Detection accuracy at IoU 0.7
- **GPU Utilization**: Average GPU usage percentage
- **Memory Usage**: Peak GPU and system memory

### Pipeline Breakdown
1. **Stereo Processing**: ~40-60% of total time
2. **Point Cloud Generation**: ~10-15% of total time
3. **3D Detection**: ~25-35% of total time
4. **Post-processing**: ~5-10% of total time

### Expected Performance Ranges

#### Jetson Orin NX 16GB
- **End-to-End FPS**: 8-15 FPS (FP32), 12-20 FPS (INT8)
- **GPU Utilization**: 85-95%
- **Memory Usage**: 4-6 GB GPU, 3-4 GB system
- **Power**: 20-35W during inference

#### Jetson Orin NX 8GB
- **End-to-End FPS**: 6-12 FPS (FP32), 10-16 FPS (INT8)
- **GPU Utilization**: 90-98%
- **Memory Usage**: 3-5 GB GPU, 2.5-3.5 GB system
- **Power**: 18-30W during inference

## Configuration Options

### Environment Variables
```bash
# Customize benchmark behavior
export DETECTION_BATCH_SIZE=2           # Batch size for inference
export DETECTION_PRECISION="int8"       # Use INT8 quantization
export DETECTION_NUM_SAMPLES=500        # Number of test samples
export TENSORRT_WORKSPACE_SIZE=4096     # TensorRT workspace in MB
export ENABLE_PROFILING=1               # Detailed TensorRT profiling
```

### Model Configuration
Edit the configuration in `run_benchmark.sh`:
```bash
# Model variants to test
PRECISIONS=("fp32" "int8")
BATCH_SIZES=(1 2 4)
OPTIMIZATION_LEVELS=("default" "aggressive")
```

## Troubleshooting

### Common Issues

#### GPU Memory Errors
- **Reduce batch size**: Set `DETECTION_BATCH_SIZE=1`
- **Check memory usage**: Monitor with `nvidia-smi`
- **Clear cache**: Restart to clear GPU memory

#### TensorRT Build Failures
```bash
# Rebuild TensorRT engines
rm -rf tensorrt_engines/
REBUILD_ENGINES=1 ./run_benchmark.sh
```

#### Low Performance
- **Check thermal throttling**: Monitor GPU temperatures
- **Verify power mode**: Ensure maximum performance mode
- **Check clock frequencies**: Use `jetson_clocks` utility

#### Accuracy Issues
```bash
# Verify model integrity
python3 scripts/validate_models.py

# Check calibration data
ls calibration_data/
```

### Performance Debugging
```bash
# Run with TensorRT profiling
ENABLE_PROFILING=1 ./run_benchmark.sh

# Monitor system resources
sudo tegrastats  # Jetson-specific monitoring
```

## Scientific Methodology

### Measurement Protocol
1. **Thermal Stabilization**: 10-minute idle period
2. **Engine Warm-up**: 50 inference iterations
3. **Measurement Phase**: 200+ samples with timing
4. **Accuracy Validation**: Comparison with reference results

### Quantization Calibration
- **Calibration Dataset**: 100 representative KITTI samples
- **Calibration Method**: Entropy-based PTQ
- **Validation**: Accuracy comparison with FP32 baseline

### Statistical Analysis
- Latency percentiles (P50, P95, P99)
- Throughput confidence intervals
- Performance stability metrics
- Thermal impact analysis

## Advanced Usage

### Custom Models
```bash
# Use custom ONNX models
export CRESTEREO_MODEL_PATH="/path/to/custom_crestereo.onnx"
export POINTPILLARS_MODEL_PATH="/path/to/custom_pointpillars.onnx"
./run_benchmark.sh
```

### Multi-Precision Comparison
```bash
# Compare all precision modes
python3 ../../../analysis/scripts/compare_precision_modes.py \
    results/run_*_fp32/ \
    results/run_*_int8/
```

### Power-Performance Analysis
```bash
# Enable power measurement
export ENABLE_POWER_MEASUREMENT=1
export POWER_METER_IP="192.168.1.100"
./run_benchmark.sh

# Analyze power-performance trade-offs
python3 ../../../analysis/scripts/power_performance_analysis.py results/
```

### Custom Evaluation
```bash
# Run on custom dataset
export CUSTOM_DATASET_PATH="/path/to/dataset"
export CUSTOM_ANNOTATIONS_PATH="/path/to/annotations"
./run_benchmark.sh --custom-dataset
```

## Model Details

### CREStereo
- **Architecture**: Recurrent stereo matching network
- **Input Resolution**: 1242x375 (KITTI standard)
- **Output**: Dense disparity maps
- **Optimization**: TensorRT with FP16/INT8 support

### PointPillars
- **Architecture**: Pillar-based 3D object detection
- **Point Cloud**: Pseudo-LiDAR from stereo depth
- **Classes**: Car, Pedestrian, Cyclist
- **NMS**: CUDA-accelerated post-processing

## References

- [PointPillars Paper](https://arxiv.org/abs/1812.05784)
- [CREStereo Paper](https://arxiv.org/abs/2203.11483)
- [KITTI 3D Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/)

## Support

For issues specific to this benchmark:
1. Check the troubleshooting section above
2. Review TensorRT installation in `../setup/README.md`
3. Consult the main project documentation in `../../../docs/`
4. Check GPU compatibility with `nvidia-smi`
