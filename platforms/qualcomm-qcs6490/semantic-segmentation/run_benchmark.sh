#!/bin/bash

# Semantic Segmentation Benchmark Script for Qualcomm QCS6490
# Implements the DDRNet-23-slim benchmark using SNPE SDK

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
DDRNET_DLC_FP32="$MODELS_PATH/dlc/ddrnet_fp32.dlc"
DDRNET_DLC_QUANTIZED="$MODELS_PATH/dlc/ddrnet_quantized.dlc"

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Cityscapes dataset
    if [ ! -d "$CITYSCAPES_DATASET_PATH" ]; then
        error "Cityscapes dataset not found at $CITYSCAPES_DATASET_PATH"
    fi
    
    # Check SNPE SDK
    if [ ! -d "$SNPE_ROOT" ]; then
        error "SNPE SDK not found. Please install SNPE SDK first."
    fi
    
    # Check SNPE tools
    if [ ! -f "$SNPE_ROOT/bin/aarch64-linux-gcc7.5/snpe-onnx-to-dlc" ]; then
        error "SNPE tools not found. Please check SNPE SDK installation."
    fi
    
    # Check if models directory exists
    mkdir -p "$MODELS_PATH/onnx"
    mkdir -p "$MODELS_PATH/dlc"
    
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
import numpy as np
import cv2
from pathlib import Path

def preprocess_and_save_raw(image_path, output_path):
    """Preprocess image and save as raw binary file for SNPE calibration."""
    # Load and preprocess image
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    
    # Resize to model input size
    image = cv2.resize(image, (2048, 1024))
    
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
    
    # Save as raw binary file
    image.astype(np.float32).tofile(str(output_path))
    
    return True

def create_calibration_subset():
    """Create a 500-image calibration subset from Cityscapes validation set."""
    cityscapes_path = Path(os.environ.get('CITYSCAPES_DATASET_PATH', '.'))
    calibration_path = Path(os.environ.get('RESULTS_DIR', '.')) / 'calibration'
    
    # Source directories
    val_images_dir = cityscapes_path / 'leftImg8bit' / 'val'
    
    if not val_images_dir.exists():
        print(f"Error: Cityscapes validation images not found at {val_images_dir}")
        return False
    
    # Create calibration directories
    (calibration_path / 'images').mkdir(parents=True, exist_ok=True)
    (calibration_path / 'raw').mkdir(parents=True, exist_ok=True)
    
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
    
    print(f"Processing {len(selected_images)} images for calibration")
    
    # Process selected images
    calibration_list = []
    for i, img_path in enumerate(selected_images):
        if i % 50 == 0:
            print(f"Processing image {i+1}/{len(selected_images)}")
        
        # Copy original image
        dst_img = calibration_path / 'images' / img_path.name
        shutil.copy2(img_path, dst_img)
        
        # Create preprocessed raw file
        raw_filename = img_path.stem + '.raw'
        raw_path = calibration_path / 'raw' / raw_filename
        
        if preprocess_and_save_raw(img_path, raw_path):
            calibration_list.append(str(raw_path.absolute()))
    
    # Create calibration file list for SNPE
    with open(calibration_path / 'calibration_list.txt', 'w') as f:
        for raw_path in calibration_list:
            f.write(f"{raw_path}\n")
    
    print(f"Created calibration dataset with {len(calibration_list)} samples")
    return True

if __name__ == "__main__":
    create_calibration_subset()
EOF
    
    CITYSCAPES_DATASET_PATH="$CITYSCAPES_DATASET_PATH" RESULTS_DIR="$RESULTS_DIR" python3 "$RESULTS_DIR/create_calibration.py"
}

# Convert DDRNet to SNPE DLC format
convert_model_to_dlc() {
    log "Converting DDRNet ONNX model to SNPE DLC format..."
    
    if [ ! -f "$DDRNET_ONNX" ]; then
        error "DDRNet ONNX model not found at $DDRNET_ONNX"
    fi
    
    # Convert ONNX to FP32 DLC
    log "Converting ONNX to FP32 DLC..."
    $SNPE_ROOT/bin/aarch64-linux-gcc7.5/snpe-onnx-to-dlc \
        --input_network "$DDRNET_ONNX" \
        --output_path "$DDRNET_DLC_FP32"
    
    if [ ! -f "$DDRNET_DLC_FP32" ]; then
        error "Failed to create FP32 DLC file"
    fi
    
    success "FP32 DLC created successfully"
    
    # Quantize to INT8 DLC for Hexagon NPU
    local calibration_list="$RESULTS_DIR/calibration/calibration_list.txt"
    
    if [ ! -f "$calibration_list" ]; then
        error "Calibration list not found. Please run calibration dataset creation first."
    fi
    
    log "Quantizing DLC for Hexagon NPU..."
    $SNPE_ROOT/bin/aarch64-linux-gcc7.5/snpe-dlc-quantize \
        --input_dlc "$DDRNET_DLC_FP32" \
        --input_list "$calibration_list" \
        --output_dlc "$DDRNET_DLC_QUANTIZED" \
        --enable_htp
    
    if [ ! -f "$DDRNET_DLC_QUANTIZED" ]; then
        error "Failed to create quantized DLC file"
    fi
    
    success "Quantized DLC created successfully"
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
from pathlib import Path
import json
import sys

# Add SNPE Python path
snpe_root = os.environ.get('SNPE_ROOT', '')
if snpe_root:
    sys.path.append(os.path.join(snpe_root, 'lib', 'python'))

try:
    import snpe
    from snpe import zdl
except ImportError:
    print("Error: SNPE Python bindings not found. Please check SNPE installation.")
    sys.exit(1)

class SNPEInference:
    def __init__(self, dlc_path, runtime='DSP'):
        self.dlc_path = dlc_path
        self.runtime = runtime
        self.snpe_net = None
        self.input_tensor = None
        self.output_tensor = None
        
        self._initialize_snpe()
    
    def _initialize_snpe(self):
        """Initialize SNPE network."""
        # Set runtime
        if self.runtime == 'DSP':
            runtime_target = zdl.DlSystem.Runtime_t.DSP
        elif self.runtime == 'GPU':
            runtime_target = zdl.DlSystem.Runtime_t.GPU
        else:
            runtime_target = zdl.DlSystem.Runtime_t.CPU
        
        # Build SNPE network
        snpe_builder = zdl.SNPE.SnpeBuilder(self.dlc_path)
        snpe_builder.setRuntime(runtime_target)
        snpe_builder.setUseUserSuppliedBuffers(False)
        
        self.snpe_net = snpe_builder.build()
        
        if not self.snpe_net:
            raise RuntimeError("Failed to build SNPE network")
        
        # Get input/output tensor names
        input_tensor_names = self.snpe_net.getInputTensorNames()
        output_tensor_names = self.snpe_net.getOutputTensorNames()
        
        if not input_tensor_names or not output_tensor_names:
            raise RuntimeError("Failed to get tensor names")
        
        self.input_tensor_name = input_tensor_names.at(0)
        self.output_tensor_name = output_tensor_names.at(0)
        
        print(f"SNPE network initialized with runtime: {self.runtime}")
        print(f"Input tensor: {self.input_tensor_name}")
        print(f"Output tensor: {self.output_tensor_name}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for DDRNet inference."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize image
        image = cv2.resize(image, (2048, 1024))
        
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
        
        return image
    
    def infer(self, input_data):
        """Run inference on input data."""
        # Create input tensor map
        input_tensor_map = zdl.TensorMap()
        
        # Create tensor from numpy array
        input_tensor = zdl.Tensor.createFloat32Tensor(input_data.shape)
        input_tensor.copyFrom(input_data)
        
        input_tensor_map.add(self.input_tensor_name, input_tensor)
        
        # Create output tensor map
        output_tensor_map = zdl.TensorMap()
        
        # Execute network
        success = self.snpe_net.execute(input_tensor_map, output_tensor_map)
        
        if not success:
            raise RuntimeError("SNPE inference failed")
        
        # Get output tensor
        output_tensor = output_tensor_map.getTensor(self.output_tensor_name)
        output_data = np.array(output_tensor)
        
        return output_data

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

def benchmark_segmentation(dlc_path, cityscapes_path, runtime='DSP', num_iterations=1000, warmup_iterations=100):
    """Benchmark DDRNet semantic segmentation."""
    
    # Initialize SNPE inference
    print("Initializing SNPE inference engine...")
    inference_engine = SNPEInference(dlc_path, runtime)
    
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
    sample_input = inference_engine.preprocess_image(image_files[0])
    
    # Warmup phase
    print(f"Running {warmup_iterations} warmup iterations...")
    for _ in range(warmup_iterations):
        _ = inference_engine.infer(sample_input)
    
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
        input_tensor = inference_engine.preprocess_image(image_path)
        
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

def analyze_results(latencies, predictions, cityscapes_path, runtime):
    """Analyze and save benchmark results."""
    latencies_array = np.array(latencies)
    
    results = {
        'runtime': runtime,
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
    
    # Save results
    results_file = f"segmentation_results_{runtime.lower()}_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Semantic Segmentation Benchmark Results - {runtime}")
    print(f"{'='*60}")
    print(f"Iterations: {results['num_iterations']}")
    print(f"Mean Latency: {results['latency']['mean_ms']:.2f} ± {results['latency']['std_ms']:.2f} ms")
    print(f"P99 Latency:  {results['latency']['p99_ms']:.2f} ms")
    print(f"P95 Latency:  {results['latency']['p95_ms']:.2f} ms")
    print(f"Throughput:   {results['throughput_fps']:.2f} FPS")
    print(f"{'='*60}")
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='DDRNet Segmentation Benchmark')
    parser.add_argument('--dlc', required=True, help='Path to DLC model file')
    parser.add_argument('--dataset', required=True, help='Path to Cityscapes dataset')
    parser.add_argument('--runtime', default='DSP', choices=['DSP', 'GPU', 'CPU'], help='SNPE runtime')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of benchmark iterations')
    parser.add_argument('--warmup', type=int, default=100, help='Number of warmup iterations')
    
    args = parser.parse_args()
    
    # Run benchmark
    latencies, predictions = benchmark_segmentation(
        args.dlc,
        args.dataset,
        args.runtime,
        args.iterations,
        args.warmup
    )
    
    # Analyze results
    analyze_results(latencies, predictions, args.dataset, args.runtime)

if __name__ == "__main__":
    main()
EOF
    
    chmod +x "$RESULTS_DIR/benchmark_segmentation.py"
}

# Run benchmark
run_benchmark() {
    log "Running semantic segmentation benchmark..."
    
    cd "$RESULTS_DIR"
    
    python3 benchmark_segmentation.py \
        --dlc "$DDRNET_DLC_QUANTIZED" \
        --dataset "$CITYSCAPES_DATASET_PATH" \
        --runtime "DSP" \
        --iterations "$NUM_ITERATIONS" \
        --warmup "$WARMUP_ITERATIONS" > "$RESULTS_DIR/logs/dsp_benchmark.log" 2>&1
    
    success "Benchmark completed"
}

# Main execution
main() {
    log "Starting Semantic Segmentation benchmark for Qualcomm QCS6490..."
    
    check_prerequisites
    setup_results_dir
    download_model
    create_calibration_dataset
    convert_model_to_dlc
    create_benchmark_script
    run_benchmark
    
    success "Semantic Segmentation benchmark completed successfully!"
    
    echo ""
    echo "=================================================="
    echo "Benchmark Complete!"
    echo "=================================================="
    echo ""
    echo "Results location: $RESULTS_DIR"
    echo ""
    echo "Key files:"
    echo "- segmentation_results_dsp_*.json: Hexagon NPU benchmark results"
    echo "- logs/: Detailed execution logs"
    echo ""
    echo "To view results:"
    echo "ls $RESULTS_DIR/*.json"
    echo ""
}

# Execute main function
main "$@"
