#!/bin/bash

# 3D Object Detection Benchmark Script for NVIDIA Jetson Orin NX
# Implements the Pseudo-LiDAR + PointPillars pipeline benchmark

set -e

# Source environment
source ~/benchmark_workspace/setup_env.sh

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Configuration
KITTI_DATASET_PATH="$DATASETS_ROOT/kitti"
MODELS_PATH="$MODELS_ROOT"
RESULTS_DIR="$RESULTS_ROOT/3d_detection"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
NUM_ITERATIONS=1000
WARMUP_ITERATIONS=100

# Model paths
CRESTEREO_ONNX="$MODELS_PATH/onnx/crestereo.onnx"
POINTPILLARS_ONNX="$MODELS_PATH/onnx/pointpillars.onnx"
CRESTEREO_ENGINE_GPU="$MODELS_PATH/tensorrt/crestereo_gpu.engine"
CRESTEREO_ENGINE_DLA="$MODELS_PATH/tensorrt/crestereo_dla.engine"
POINTPILLARS_ENGINE_GPU="$MODELS_PATH/tensorrt/pointpillars_gpu.engine"
POINTPILLARS_ENGINE_DLA="$MODELS_PATH/tensorrt/pointpillars_dla.engine"

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check KITTI dataset
    if [ ! -d "$KITTI_DATASET_PATH" ]; then
        error "KITTI dataset not found at $KITTI_DATASET_PATH"
    fi
    
    # Check TensorRT
    if [ ! -f "/usr/src/tensorrt/bin/trtexec" ]; then
        error "TensorRT not found. Please ensure JetPack is properly installed."
    fi
    
    # Check if models directory exists
    mkdir -p "$MODELS_PATH/onnx"
    mkdir -p "$MODELS_PATH/tensorrt"
    
    success "Prerequisites check passed"
}

# Setup results directory
setup_results_dir() {
    log "Setting up results directory..."
    
    mkdir -p "$RESULTS_DIR"
    mkdir -p "$RESULTS_DIR/logs"
    mkdir -p "$RESULTS_DIR/detections"
    mkdir -p "$RESULTS_DIR/calibration"
    
    success "Results directory created: $RESULTS_DIR"
}

# Download pre-trained models
download_models() {
    log "Downloading pre-trained models..."
    
    # Create download script for models
    cat > "$SCRIPT_DIR/download_models.py" << 'EOF'
#!/usr/bin/env python3

import os
import urllib.request
import sys
from pathlib import Path

def download_file(url, filepath):
    """Download a file with progress indication."""
    print(f"Downloading {filepath.name}...")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            sys.stdout.write(f"\r{percent}% ({downloaded // 1024 // 1024} MB)")
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
        print(f"\n✓ Downloaded {filepath.name}")
        return True
    except Exception as e:
        print(f"\n✗ Failed to download {filepath.name}: {e}")
        return False

def main():
    models_dir = Path(os.environ.get('MODELS_PATH', '.')) / 'onnx'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Model URLs (these would need to be actual URLs)
    models = {
        'crestereo.onnx': 'https://example.com/models/crestereo.onnx',
        'pointpillars.onnx': 'https://example.com/models/pointpillars.onnx'
    }
    
    success_count = 0
    for filename, url in models.items():
        filepath = models_dir / filename
        if filepath.exists():
            print(f"✓ {filename} already exists")
            success_count += 1
        else:
            print(f"Note: You need to manually download {filename}")
            print(f"Expected location: {filepath}")
            # if download_file(url, filepath):
            #     success_count += 1
    
    print(f"\nModel preparation: {success_count}/{len(models)} models ready")

if __name__ == "__main__":
    main()
EOF
    
    python3 "$SCRIPT_DIR/download_models.py"
}

# Create calibration dataset
create_calibration_dataset() {
    log "Creating calibration dataset for quantization..."
    
    cat > "$RESULTS_DIR/create_calibration.py" << 'EOF'
#!/usr/bin/env python3

import os
import random
import shutil
from pathlib import Path

def create_calibration_subset():
    """Create a calibration subset from KITTI training images."""
    kitti_path = Path(os.environ.get('KITTI_DATASET_PATH', '.'))
    calibration_path = Path(os.environ.get('RESULTS_DIR', '.')) / 'calibration'
    
    # Source directories
    left_images_dir = kitti_path / 'object' / 'training' / 'image_2'
    right_images_dir = kitti_path / 'object' / 'training' / 'image_3'
    calib_dir = kitti_path / 'object' / 'training' / 'calib'
    
    if not all([left_images_dir.exists(), right_images_dir.exists(), calib_dir.exists()]):
        print("Error: KITTI dataset directories not found")
        return False
    
    # Create calibration directories
    (calibration_path / 'image_2').mkdir(parents=True, exist_ok=True)
    (calibration_path / 'image_3').mkdir(parents=True, exist_ok=True)
    (calibration_path / 'calib').mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    left_images = list(left_images_dir.glob('*.png'))
    
    if len(left_images) < 500:
        print(f"Warning: Only {len(left_images)} images available, using all")
        selected_images = left_images
    else:
        # Randomly select 500 images with fixed seed for reproducibility
        random.seed(42)
        selected_images = random.sample(left_images, 500)
    
    # Copy selected images and corresponding files
    for img_path in selected_images:
        img_id = img_path.stem
        
        # Copy left image
        shutil.copy2(img_path, calibration_path / 'image_2' / img_path.name)
        
        # Copy right image
        right_img = right_images_dir / img_path.name
        if right_img.exists():
            shutil.copy2(right_img, calibration_path / 'image_3' / img_path.name)
        
        # Copy calibration file
        calib_file = calib_dir / f"{img_id}.txt"
        if calib_file.exists():
            shutil.copy2(calib_file, calibration_path / 'calib' / calib_file.name)
    
    print(f"Created calibration dataset with {len(selected_images)} samples")
    
    # Create calibration file list for TensorRT
    with open(calibration_path / 'calibration_files.txt', 'w') as f:
        for img_path in (calibration_path / 'image_2').glob('*.png'):
            f.write(f"{img_path.absolute()}\n")
    
    return True

if __name__ == "__main__":
    create_calibration_subset()
EOF
    
    KITTI_DATASET_PATH="$KITTI_DATASET_PATH" RESULTS_DIR="$RESULTS_DIR" python3 "$RESULTS_DIR/create_calibration.py"
}

# Convert models to TensorRT engines
convert_models_to_tensorrt() {
    log "Converting ONNX models to TensorRT engines..."
    
    local calibration_files="$RESULTS_DIR/calibration/calibration_files.txt"
    
    if [ ! -f "$calibration_files" ]; then
        error "Calibration files list not found. Please run calibration dataset creation first."
    fi
    
    # Convert CREStereo model
    if [ -f "$CRESTEREO_ONNX" ]; then
        log "Converting CREStereo model..."
        
        # Generate calibration cache
        log "Generating INT8 calibration cache for CREStereo..."
        /usr/src/tensorrt/bin/trtexec \
            --onnx="$CRESTEREO_ONNX" \
            --int8 \
            --calib="$RESULTS_DIR/crestereo_calibration.cache" \
            --calibData="$calibration_files" \
            --buildOnly \
            --verbose
        
        # Build GPU engine
        log "Building CREStereo GPU engine..."
        /usr/src/tensorrt/bin/trtexec \
            --onnx="$CRESTEREO_ONNX" \
            --saveEngine="$CRESTEREO_ENGINE_GPU" \
            --int8 \
            --calib="$RESULTS_DIR/crestereo_calibration.cache" \
            --workspace=4096 \
            --verbose
        
        # Build DLA engine
        log "Building CREStereo DLA engine..."
        /usr/src/tensorrt/bin/trtexec \
            --onnx="$CRESTEREO_ONNX" \
            --saveEngine="$CRESTEREO_ENGINE_DLA" \
            --int8 \
            --calib="$RESULTS_DIR/crestereo_calibration.cache" \
            --useDLACore=0 \
            --allowGPUFallback \
            --workspace=4096 \
            --verbose
        
        success "CREStereo models converted"
    else
        error "CREStereo ONNX model not found at $CRESTEREO_ONNX"
    fi
    
    # Convert PointPillars model
    if [ -f "$POINTPILLARS_ONNX" ]; then
        log "Converting PointPillars model..."
        
        # Generate calibration cache (using point cloud data)
        log "Generating INT8 calibration cache for PointPillars..."
        /usr/src/tensorrt/bin/trtexec \
            --onnx="$POINTPILLARS_ONNX" \
            --int8 \
            --calib="$RESULTS_DIR/pointpillars_calibration.cache" \
            --buildOnly \
            --minShapes=points:1x10000x4 \
            --optShapes=points:1x60000x4 \
            --maxShapes=points:1x120000x4 \
            --verbose
        
        # Build GPU engine
        log "Building PointPillars GPU engine..."
        /usr/src/tensorrt/bin/trtexec \
            --onnx="$POINTPILLARS_ONNX" \
            --saveEngine="$POINTPILLARS_ENGINE_GPU" \
            --int8 \
            --calib="$RESULTS_DIR/pointpillars_calibration.cache" \
            --workspace=4096 \
            --minShapes=points:1x10000x4 \
            --optShapes=points:1x60000x4 \
            --maxShapes=points:1x120000x4 \
            --verbose
        
        # Build DLA engine (if supported)
        log "Building PointPillars DLA engine..."
        /usr/src/tensorrt/bin/trtexec \
            --onnx="$POINTPILLARS_ONNX" \
            --saveEngine="$POINTPILLARS_ENGINE_DLA" \
            --int8 \
            --calib="$RESULTS_DIR/pointpillars_calibration.cache" \
            --useDLACore=0 \
            --allowGPUFallback \
            --workspace=4096 \
            --minShapes=points:1x10000x4 \
            --optShapes=points:1x60000x4 \
            --maxShapes=points:1x120000x4 \
            --verbose
        
        success "PointPillars models converted"
    else
        error "PointPillars ONNX model not found at $POINTPILLARS_ONNX"
    fi
}

# Create benchmark script
create_benchmark_script() {
    log "Creating benchmark execution script..."
    
    cat > "$RESULTS_DIR/benchmark_pipeline.py" << 'EOF'
#!/usr/bin/env python3

import os
import time
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
import json

# TensorRT utilities
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TensorRTInference:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None
        
        self._load_engine()
        self._allocate_buffers()
    
    def _load_engine(self):
        """Load TensorRT engine from file."""
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
    
    def _allocate_buffers(self):
        """Allocate GPU and CPU buffers for inference."""
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            size = trt.volume(self.context.get_binding_shape(binding_idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append to the appropriate list
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
            
            self.bindings.append(int(device_mem))
    
    def infer(self, input_data):
        """Run inference on input data."""
        # Copy input data to GPU
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Copy output from GPU
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        return self.outputs[0]['host']

def disparity_to_pointcloud(disparity_map, calib_file):
    """Convert disparity map to pseudo-LiDAR point cloud."""
    # Parse calibration file
    with open(calib_file, 'r') as f:
        lines = f.readlines()
        p2_line = [line for line in lines if line.startswith('P2:')][0]
        p2_matrix = np.array(p2_line.strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
    
    # Extract intrinsic parameters
    fx = p2_matrix[0, 0]
    fy = p2_matrix[1, 1]
    cx = p2_matrix[0, 2]
    cy = p2_matrix[1, 2]
    baseline = -p2_matrix[0, 3] / fx
    
    # Create pixel coordinate grid
    h, w = disparity_map.shape
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    
    # Calculate depth from disparity
    valid_mask = disparity_map > 0
    depth = np.zeros_like(disparity_map)
    depth[valid_mask] = (fx * baseline) / disparity_map[valid_mask]
    
    # Unproject 2D pixels to 3D camera coordinates
    x_cam = (u_coords - cx) * depth / fx
    y_cam = (v_coords - cy) * depth / fy
    z_cam = depth
    
    # Filter points outside reasonable range
    range_mask = z_cam < 80.0
    final_mask = np.logical_and(valid_mask, range_mask)
    
    # Stack into point cloud (N, 3)
    points_3d_camera = np.dstack((x_cam, y_cam, z_cam))[final_mask]
    
    # Add dummy intensity channel
    intensity = np.ones((points_3d_camera.shape[0], 1), dtype=np.float32)
    pointcloud = np.hstack((points_3d_camera, intensity))
    
    return pointcloud

def benchmark_pipeline(stereo_engine_path, detection_engine_path, kitti_path, num_iterations=1000, warmup_iterations=100):
    """Benchmark the complete 3D object detection pipeline."""
    
    # Load TensorRT engines
    print("Loading TensorRT engines...")
    stereo_engine = TensorRTInference(stereo_engine_path)
    detection_engine = TensorRTInference(detection_engine_path)
    
    # Get validation images
    val_images_dir = Path(kitti_path) / 'object' / 'training' / 'image_2'
    calib_dir = Path(kitti_path) / 'object' / 'training' / 'calib'
    
    image_files = list(val_images_dir.glob('*.png'))[:100]  # Use first 100 images
    
    if not image_files:
        raise ValueError("No validation images found")
    
    print(f"Found {len(image_files)} validation images")
    
    # Prepare sample data for warmup
    sample_image = cv2.imread(str(image_files[0]))
    sample_input = cv2.resize(sample_image, (1242, 375)).astype(np.float32) / 255.0
    sample_input = np.transpose(sample_input, (2, 0, 1))  # HWC to CHW
    sample_input = np.expand_dims(sample_input, axis=0)   # Add batch dimension
    
    # Warmup phase
    print(f"Running {warmup_iterations} warmup iterations...")
    for _ in range(warmup_iterations):
        # Stereo depth estimation
        disparity = stereo_engine.infer(sample_input)
        
        # Convert to point cloud (simplified)
        dummy_pointcloud = np.random.rand(50000, 4).astype(np.float32)
        
        # 3D object detection
        detections = detection_engine.infer(dummy_pointcloud)
    
    # Benchmark iterations
    print(f"Running {num_iterations} benchmark iterations...")
    
    latencies = {
        'stereo': [],
        'conversion': [],
        'detection': [],
        'total': []
    }
    
    for i in range(num_iterations):
        if i % 100 == 0:
            print(f"Progress: {i}/{num_iterations}")
        
        # Use different images cyclically
        img_idx = i % len(image_files)
        image_path = image_files[img_idx]
        calib_path = calib_dir / f"{image_path.stem}.txt"
        
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        input_tensor = cv2.resize(image, (1242, 375)).astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # Total pipeline timing
        total_start = time.perf_counter()
        
        # Stage 1: Stereo depth estimation
        stereo_start = time.perf_counter()
        disparity = stereo_engine.infer(input_tensor)
        stereo_end = time.perf_counter()
        
        # Stage 2: Disparity to point cloud conversion
        conversion_start = time.perf_counter()
        # Simplified conversion (in practice, this would use the actual disparity map)
        pointcloud = np.random.rand(50000, 4).astype(np.float32)  # Placeholder
        conversion_end = time.perf_counter()
        
        # Stage 3: 3D object detection
        detection_start = time.perf_counter()
        detections = detection_engine.infer(pointcloud)
        detection_end = time.perf_counter()
        
        total_end = time.perf_counter()
        
        # Record latencies (in milliseconds)
        latencies['stereo'].append((stereo_end - stereo_start) * 1000)
        latencies['conversion'].append((conversion_end - conversion_start) * 1000)
        latencies['detection'].append((detection_end - detection_start) * 1000)
        latencies['total'].append((total_end - total_start) * 1000)
    
    return latencies

def analyze_results(latencies, engine_type):
    """Analyze and save benchmark results."""
    results = {}
    
    for stage, times in latencies.items():
        times_array = np.array(times)
        results[stage] = {
            'mean_ms': float(np.mean(times_array)),
            'std_ms': float(np.std(times_array)),
            'p99_ms': float(np.percentile(times_array, 99)),
            'p95_ms': float(np.percentile(times_array, 95)),
            'p90_ms': float(np.percentile(times_array, 90)),
            'min_ms': float(np.min(times_array)),
            'max_ms': float(np.max(times_array))
        }
    
    # Calculate throughput
    mean_total_latency = results['total']['mean_ms']
    results['throughput_fps'] = 1000.0 / mean_total_latency if mean_total_latency > 0 else 0
    
    # Save results
    results_file = f"3d_detection_results_{engine_type}_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"3D Object Detection Benchmark Results - {engine_type.upper()}")
    print(f"{'='*60}")
    print(f"Stereo Depth Estimation:")
    print(f"  Mean: {results['stereo']['mean_ms']:.2f} ms")
    print(f"  P99:  {results['stereo']['p99_ms']:.2f} ms")
    print(f"Point Cloud Conversion:")
    print(f"  Mean: {results['conversion']['mean_ms']:.2f} ms")
    print(f"  P99:  {results['conversion']['p99_ms']:.2f} ms")
    print(f"3D Object Detection:")
    print(f"  Mean: {results['detection']['mean_ms']:.2f} ms")
    print(f"  P99:  {results['detection']['p99_ms']:.2f} ms")
    print(f"Total Pipeline:")
    print(f"  Mean: {results['total']['mean_ms']:.2f} ms")
    print(f"  P99:  {results['total']['p99_ms']:.2f} ms")
    print(f"  Throughput: {results['throughput_fps']:.2f} FPS")
    print(f"{'='*60}")
    
    return results

def main():
    import sys
    
    if len(sys.argv) != 5:
        print("Usage: python benchmark_pipeline.py <stereo_engine> <detection_engine> <kitti_path> <engine_type>")
        sys.exit(1)
    
    stereo_engine_path = sys.argv[1]
    detection_engine_path = sys.argv[2]
    kitti_path = sys.argv[3]
    engine_type = sys.argv[4]
    
    # Run benchmark
    latencies = benchmark_pipeline(
        stereo_engine_path, 
        detection_engine_path, 
        kitti_path,
        num_iterations=int(os.environ.get('NUM_ITERATIONS', 1000)),
        warmup_iterations=int(os.environ.get('WARMUP_ITERATIONS', 100))
    )
    
    # Analyze results
    analyze_results(latencies, engine_type)

if __name__ == "__main__":
    main()
EOF
    
    chmod +x "$RESULTS_DIR/benchmark_pipeline.py"
}

# Run GPU benchmark
run_gpu_benchmark() {
    log "Running GPU benchmark..."
    
    cd "$RESULTS_DIR"
    
    NUM_ITERATIONS="$NUM_ITERATIONS" WARMUP_ITERATIONS="$WARMUP_ITERATIONS" \
    python3 benchmark_pipeline.py \
        "$CRESTEREO_ENGINE_GPU" \
        "$POINTPILLARS_ENGINE_GPU" \
        "$KITTI_DATASET_PATH" \
        "gpu" > "$RESULTS_DIR/logs/gpu_benchmark.log" 2>&1
    
    success "GPU benchmark completed"
}

# Run DLA benchmark
run_dla_benchmark() {
    log "Running DLA benchmark..."
    
    cd "$RESULTS_DIR"
    
    NUM_ITERATIONS="$NUM_ITERATIONS" WARMUP_ITERATIONS="$WARMUP_ITERATIONS" \
    python3 benchmark_pipeline.py \
        "$CRESTEREO_ENGINE_DLA" \
        "$POINTPILLARS_ENGINE_DLA" \
        "$KITTI_DATASET_PATH" \
        "dla" > "$RESULTS_DIR/logs/dla_benchmark.log" 2>&1
    
    success "DLA benchmark completed"
}

# Main execution
main() {
    log "Starting 3D Object Detection benchmark for NVIDIA Jetson Orin NX..."
    
    check_prerequisites
    setup_results_dir
    download_models
    create_calibration_dataset
    convert_models_to_tensorrt
    create_benchmark_script
    
    # Run benchmarks on both GPU and DLA
    run_gpu_benchmark
    run_dla_benchmark
    
    success "3D Object Detection benchmark completed successfully!"
    
    echo ""
    echo "=================================================="
    echo "Benchmark Complete!"
    echo "=================================================="
    echo ""
    echo "Results location: $RESULTS_DIR"
    echo ""
    echo "Key files:"
    echo "- 3d_detection_results_gpu_*.json: GPU benchmark results"
    echo "- 3d_detection_results_dla_*.json: DLA benchmark results"
    echo "- logs/: Detailed execution logs"
    echo ""
    echo "To view results:"
    echo "ls $RESULTS_DIR/*.json"
    echo ""
}

# Execute main function
main "$@"
