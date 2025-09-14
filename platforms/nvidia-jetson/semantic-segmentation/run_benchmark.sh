#!/bin/bash

# Semantic Segmentation Benchmark Script for NVIDIA Jetson Orin NX
# Implements the DDRNet-23-slim benchmark

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
CITYSCAPES_DATASET_PATH="$DATASETS_ROOT/cityscapes"
MODELS_PATH="$MODELS_ROOT"
RESULTS_DIR="$RESULTS_ROOT/segmentation"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
NUM_ITERATIONS=1000
WARMUP_ITERATIONS=100

# Model paths
DDRNET_ONNX="$MODELS_PATH/onnx/ddrnet23-slim.onnx"
DDRNET_ENGINE_GPU="$MODELS_PATH/tensorrt/ddrnet_gpu.engine"
DDRNET_ENGINE_DLA="$MODELS_PATH/tensorrt/ddrnet_dla.engine"

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Cityscapes dataset
    if [ ! -d "$CITYSCAPES_DATASET_PATH" ]; then
        error "Cityscapes dataset not found at $CITYSCAPES_DATASET_PATH"
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
    mkdir -p "$RESULTS_DIR/calibration"
    mkdir -p "$RESULTS_DIR/predictions"
    
    success "Results directory created: $RESULTS_DIR"
}

# Download DDRNet model
download_model() {
    log "Downloading DDRNet-23-slim model..."
    
    cat > "$SCRIPT_DIR/download_ddrnet.py" << 'EOF'
#!/usr/bin/env python3

import os
import urllib.request
import sys
from pathlib import Path

def download_ddrnet():
    """Download DDRNet-23-slim model from Hugging Face."""
    models_dir = Path(os.environ.get('MODELS_PATH', '.')) / 'onnx'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / 'ddrnet23-slim.onnx'
    
    if model_path.exists():
        print(f"✓ DDRNet model already exists at {model_path}")
        return True
    
    # Hugging Face model URL
    model_url = "https://huggingface.co/qualcomm/DDRNet23-Slim/resolve/main/DDRNet23-Slim.onnx"
    
    print(f"Downloading DDRNet model to {model_path}...")
    
    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                sys.stdout.write(f"\r{percent}% ({downloaded // 1024 // 1024} MB)")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(model_url, model_path, reporthook=progress_hook)
        print(f"\n✓ Successfully downloaded DDRNet model")
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to download DDRNet model: {e}")
        print(f"Please manually download from: {model_url}")
        print(f"Save to: {model_path}")
        return False

if __name__ == "__main__":
    download_ddrnet()
EOF
    
    MODELS_PATH="$MODELS_PATH" python3 "$SCRIPT_DIR/download_ddrnet.py"
}

# Create calibration dataset from Cityscapes
create_calibration_dataset() {
    log "Creating calibration dataset from Cityscapes..."
    
    cat > "$RESULTS_DIR/create_calibration.py" << 'EOF'
#!/usr/bin/env python3

import os
import random
import shutil
from pathlib import Path

def create_calibration_subset():
    """Create a 500-image calibration subset from Cityscapes validation set."""
    cityscapes_path = Path(os.environ.get('CITYSCAPES_DATASET_PATH', '.'))
    calibration_path = Path(os.environ.get('RESULTS_DIR', '.')) / 'calibration'
    
    # Source directories
    val_images_dir = cityscapes_path / 'leftImg8bit' / 'val'
    val_labels_dir = cityscapes_path / 'gtFine' / 'val'
    
    if not val_images_dir.exists():
        print(f"Error: Cityscapes validation images not found at {val_images_dir}")
        return False
    
    # Create calibration directories
    (calibration_path / 'images').mkdir(parents=True, exist_ok=True)
    if val_labels_dir.exists():
        (calibration_path / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Collect all validation images
    all_images = []
    for city_dir in val_images_dir.iterdir():
        if city_dir.is_dir():
            all_images.extend(list(city_dir.glob('*_leftImg8bit.png')))
    
    if len(all_images) < 500:
        print(f"Warning: Only {len(all_images)} images available, using all")
        selected_images = all_images
    else:
        # Randomly select 500 images with fixed seed for reproducibility
        random.seed(42)
        selected_images = random.sample(all_images, 500)
    
    print(f"Selected {len(selected_images)} images for calibration")
    
    # Copy selected images and labels
    for img_path in selected_images:
        # Copy image
        dst_img = calibration_path / 'images' / img_path.name
        shutil.copy2(img_path, dst_img)
        
        # Copy corresponding label if it exists
        if val_labels_dir.exists():
            label_name = img_path.name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
            city = img_path.parent.name
            label_path = val_labels_dir / city / label_name
            
            if label_path.exists():
                dst_label = calibration_path / 'labels' / label_name
                shutil.copy2(label_path, dst_label)
    
    # Create calibration file list for TensorRT
    with open(calibration_path / 'calibration_files.txt', 'w') as f:
        for img_path in (calibration_path / 'images').glob('*.png'):
            f.write(f"{img_path.absolute()}\n")
    
    print(f"Created calibration dataset with {len(selected_images)} samples")
    return True

if __name__ == "__main__":
    create_calibration_subset()
EOF
    
    CITYSCAPES_DATASET_PATH="$CITYSCAPES_DATASET_PATH" RESULTS_DIR="$RESULTS_DIR" python3 "$RESULTS_DIR/create_calibration.py"
}

# Convert DDRNet to TensorRT engines
convert_model_to_tensorrt() {
    log "Converting DDRNet ONNX model to TensorRT engines..."
    
    local calibration_files="$RESULTS_DIR/calibration/calibration_files.txt"
    
    if [ ! -f "$DDRNET_ONNX" ]; then
        error "DDRNet ONNX model not found at $DDRNET_ONNX"
    fi
    
    if [ ! -f "$calibration_files" ]; then
        error "Calibration files list not found. Please run calibration dataset creation first."
    fi
    
    # Generate calibration cache
    log "Generating INT8 calibration cache for DDRNet..."
    /usr/src/tensorrt/bin/trtexec \
        --onnx="$DDRNET_ONNX" \
        --int8 \
        --calib="$RESULTS_DIR/ddrnet_calibration.cache" \
        --calibData="$calibration_files" \
        --buildOnly \
        --verbose
    
    # Build GPU engine
    log "Building DDRNet GPU engine..."
    /usr/src/tensorrt/bin/trtexec \
        --onnx="$DDRNET_ONNX" \
        --saveEngine="$DDRNET_ENGINE_GPU" \
        --int8 \
        --calib="$RESULTS_DIR/ddrnet_calibration.cache" \
        --workspace=4096 \
        --verbose
    
    # Build DLA engine
    log "Building DDRNet DLA engine..."
    /usr/src/tensorrt/bin/trtexec \
        --onnx="$DDRNET_ONNX" \
        --saveEngine="$DDRNET_ENGINE_DLA" \
        --int8 \
        --calib="$RESULTS_DIR/ddrnet_calibration.cache" \
        --useDLACore=0 \
        --allowGPUFallback \
        --workspace=4096 \
        --verbose
    
    success "DDRNet models converted to TensorRT engines"
}

# Create benchmark script
create_benchmark_script() {
    log "Creating semantic segmentation benchmark script..."
    
    cat > "$RESULTS_DIR/benchmark_segmentation.py" << 'EOF'
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

def preprocess_image(image_path, target_size=(1024, 2048)):
    """Preprocess image for DDRNet inference."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize image
    image = cv2.resize(image, target_size)
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    
    # Convert HWC to CHW
    image = np.transpose(image, (2, 0, 1))
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def calculate_miou(pred_mask, gt_mask, num_classes=19):
    """Calculate mean Intersection over Union (mIoU)."""
    ious = []
    
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)
        
        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou)
    
    return np.mean(ious) if ious else 0.0

def benchmark_segmentation(engine_path, cityscapes_path, num_iterations=1000, warmup_iterations=100):
    """Benchmark DDRNet semantic segmentation."""
    
    # Load TensorRT engine
    print("Loading TensorRT engine...")
    inference_engine = TensorRTInference(engine_path)
    
    # Get validation images
    val_images_dir = Path(cityscapes_path) / 'leftImg8bit' / 'val'
    val_labels_dir = Path(cityscapes_path) / 'gtFine' / 'val'
    
    # Collect all validation images
    image_files = []
    for city_dir in val_images_dir.iterdir():
        if city_dir.is_dir():
            image_files.extend(list(city_dir.glob('*_leftImg8bit.png')))
    
    if not image_files:
        raise ValueError("No validation images found")
    
    print(f"Found {len(image_files)} validation images")
    
    # Prepare sample data for warmup
    sample_image = preprocess_image(image_files[0])
    
    # Warmup phase
    print(f"Running {warmup_iterations} warmup iterations...")
    for _ in range(warmup_iterations):
        _ = inference_engine.infer(sample_image)
    
    # Benchmark iterations
    print(f"Running {num_iterations} benchmark iterations...")
    
    latencies = []
    predictions = []
    
    for i in range(num_iterations):
        if i % 100 == 0:
            print(f"Progress: {i}/{num_iterations}")
        
        # Use different images cyclically
        img_idx = i % len(image_files)
        image_path = image_files[img_idx]
        
        # Preprocess image
        input_tensor = preprocess_image(image_path)
        
        # Run inference with timing
        start_time = time.perf_counter()
        output = inference_engine.infer(input_tensor)
        end_time = time.perf_counter()
        
        # Record latency (in milliseconds)
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        # Store prediction for accuracy calculation (every 10th iteration)
        if i % 10 == 0:
            predictions.append({
                'image_path': str(image_path),
                'output': output.copy()
            })
    
    return latencies, predictions

def calculate_accuracy(predictions, cityscapes_path):
    """Calculate segmentation accuracy using stored predictions."""
    val_labels_dir = Path(cityscapes_path) / 'gtFine' / 'val'
    
    if not val_labels_dir.exists():
        print("Ground truth labels not available, skipping accuracy calculation")
        return None
    
    ious = []
    
    for pred_data in predictions:
        image_path = Path(pred_data['image_path'])
        output = pred_data['output']
        
        # Get corresponding label file
        label_name = image_path.name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
        city = image_path.parent.name
        label_path = val_labels_dir / city / label_name
        
        if label_path.exists():
            # Load ground truth
            gt_mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            
            # Reshape output to mask format
            pred_mask = output.reshape((1024, 2048))  # Adjust based on actual output shape
            pred_mask = np.argmax(pred_mask, axis=0) if len(pred_mask.shape) > 2 else pred_mask
            
            # Calculate IoU
            iou = calculate_miou(pred_mask, gt_mask)
            ious.append(iou)
    
    return np.mean(ious) if ious else None

def analyze_results(latencies, predictions, cityscapes_path, engine_type):
    """Analyze and save benchmark results."""
    latencies_array = np.array(latencies)
    
    results = {
        'engine_type': engine_type,
        'num_iterations': len(latencies),
        'latency': {
            'mean_ms': float(np.mean(latencies_array)),
            'std_ms': float(np.std(latencies_array)),
            'p99_ms': float(np.percentile(latencies_array, 99)),
            'p95_ms': float(np.percentile(latencies_array, 95)),
            'p90_ms': float(np.percentile(latencies_array, 90)),
            'min_ms': float(np.min(latencies_array)),
            'max_ms': float(np.max(latencies_array))
        }
    }
    
    # Calculate throughput
    mean_latency_ms = results['latency']['mean_ms']
    results['throughput_fps'] = 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0
    
    # Calculate accuracy if possible
    accuracy = calculate_accuracy(predictions, cityscapes_path)
    if accuracy is not None:
        results['miou'] = float(accuracy)
    
    # Save results
    results_file = f"segmentation_results_{engine_type}_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Semantic Segmentation Benchmark Results - {engine_type.upper()}")
    print(f"{'='*60}")
    print(f"Iterations: {results['num_iterations']}")
    print(f"Mean Latency: {results['latency']['mean_ms']:.2f} ± {results['latency']['std_ms']:.2f} ms")
    print(f"P99 Latency:  {results['latency']['p99_ms']:.2f} ms")
    print(f"P95 Latency:  {results['latency']['p95_ms']:.2f} ms")
    print(f"Throughput:   {results['throughput_fps']:.2f} FPS")
    if 'miou' in results:
        print(f"mIoU:         {results['miou']:.4f}")
    print(f"{'='*60}")
    
    return results

def main():
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python benchmark_segmentation.py <engine_path> <cityscapes_path> <engine_type>")
        sys.exit(1)
    
    engine_path = sys.argv[1]
    cityscapes_path = sys.argv[2]
    engine_type = sys.argv[3]
    
    # Run benchmark
    latencies, predictions = benchmark_segmentation(
        engine_path,
        cityscapes_path,
        num_iterations=int(os.environ.get('NUM_ITERATIONS', 1000)),
        warmup_iterations=int(os.environ.get('WARMUP_ITERATIONS', 100))
    )
    
    # Analyze results
    analyze_results(latencies, predictions, cityscapes_path, engine_type)

if __name__ == "__main__":
    main()
EOF
    
    chmod +x "$RESULTS_DIR/benchmark_segmentation.py"
}

# Run GPU benchmark
run_gpu_benchmark() {
    log "Running GPU benchmark..."
    
    cd "$RESULTS_DIR"
    
    NUM_ITERATIONS="$NUM_ITERATIONS" WARMUP_ITERATIONS="$WARMUP_ITERATIONS" \
    python3 benchmark_segmentation.py \
        "$DDRNET_ENGINE_GPU" \
        "$CITYSCAPES_DATASET_PATH" \
        "gpu" > "$RESULTS_DIR/logs/gpu_benchmark.log" 2>&1
    
    success "GPU benchmark completed"
}

# Run DLA benchmark
run_dla_benchmark() {
    log "Running DLA benchmark..."
    
    cd "$RESULTS_DIR"
    
    NUM_ITERATIONS="$NUM_ITERATIONS" WARMUP_ITERATIONS="$WARMUP_ITERATIONS" \
    python3 benchmark_segmentation.py \
        "$DDRNET_ENGINE_DLA" \
        "$CITYSCAPES_DATASET_PATH" \
        "dla" > "$RESULTS_DIR/logs/dla_benchmark.log" 2>&1
    
    success "DLA benchmark completed"
}

# Main execution
main() {
    log "Starting Semantic Segmentation benchmark for NVIDIA Jetson Orin NX..."
    
    check_prerequisites
    setup_results_dir
    download_model
    create_calibration_dataset
    convert_model_to_tensorrt
    create_benchmark_script
    
    # Run benchmarks on both GPU and DLA
    run_gpu_benchmark
    run_dla_benchmark
    
    success "Semantic Segmentation benchmark completed successfully!"
    
    echo ""
    echo "=================================================="
    echo "Benchmark Complete!"
    echo "=================================================="
    echo ""
    echo "Results location: $RESULTS_DIR"
    echo ""
    echo "Key files:"
    echo "- segmentation_results_gpu_*.json: GPU benchmark results"
    echo "- segmentation_results_dla_*.json: DLA benchmark results"
    echo "- logs/: Detailed execution logs"
    echo ""
    echo "To view results:"
    echo "ls $RESULTS_DIR/*.json"
    echo ""
}

# Execute main function
main "$@"
