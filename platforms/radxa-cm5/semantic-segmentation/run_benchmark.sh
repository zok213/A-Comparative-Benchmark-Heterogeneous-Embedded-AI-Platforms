#!/bin/bash

# Semantic Segmentation Benchmark Script for Radxa CM5 (RK3588S)
# Implements the DDRNet-23-slim benchmark using RKNN toolkit

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
DDRNET_RKNN_NPU="$MODELS_PATH/rknn/ddrnet_npu.rknn"
DDRNET_RKNN_GPU="$MODELS_PATH/rknn/ddrnet_gpu.rknn"
DDRNET_RKNN_CPU="$MODELS_PATH/rknn/ddrnet_cpu.rknn"

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Cityscapes dataset
    if [ ! -d "$CITYSCAPES_DATASET_PATH" ]; then
        error "Cityscapes dataset not found at $CITYSCAPES_DATASET_PATH"
    fi
    
    # Check RKNN toolkit installation
    python3 -c "from rknn.api import RKNN; print('RKNN toolkit available')" || {
        error "RKNN toolkit not found. Please install RKNN toolkit first."
    }
    
    # Check Mali GPU
    if [ ! -f "/sys/class/misc/mali0/device/uevent" ]; then
        warning "Mali GPU not detected, will use NPU and CPU inference"
    fi
    
    # Check NPU
    if [ ! -f "/sys/kernel/debug/rknpu/version" ]; then
        warning "NPU not detected, will use GPU and CPU inference"
    fi
    
    # Check if models directory exists
    mkdir -p "$MODELS_PATH/onnx"
    mkdir -p "$MODELS_PATH/rknn"
    
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
    
    if not val_images_dir.exists():
        print(f"Error: Cityscapes validation images not found at {val_images_dir}")
        return False
    
    # Create calibration directories
    (calibration_path / 'images').mkdir(parents=True, exist_ok=True)
    
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
    
    # Copy selected images
    for img_path in selected_images:
        dst_img = calibration_path / 'images' / img_path.name
        shutil.copy2(img_path, dst_img)
    
    print(f"Created calibration dataset with {len(selected_images)} samples")
    return True

if __name__ == "__main__":
    create_calibration_subset()
EOF
    
    CITYSCAPES_DATASET_PATH="$CITYSCAPES_DATASET_PATH" RESULTS_DIR="$RESULTS_DIR" python3 "$RESULTS_DIR/create_calibration.py"
}

# Convert DDRNet to RKNN format
convert_model_to_rknn() {
    log "Converting DDRNet ONNX model to RKNN format..."
    
    if [ ! -f "$DDRNET_ONNX" ]; then
        error "DDRNet ONNX model not found at $DDRNET_ONNX"
    fi
    
    # Create RKNN conversion script
    cat > "$RESULTS_DIR/convert_ddrnet_to_rknn.py" << 'EOF'
#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from rknn.api import RKNN

def convert_ddrnet_to_rknn(onnx_path, rknn_path, target_platform='rk3588', quantize=True):
    """Convert DDRNet ONNX model to RKNN format."""
    
    # Create RKNN object
    rknn = RKNN(verbose=True)
    
    print(f"Converting {onnx_path} to {rknn_path}")
    
    # Config for target platform
    print(f"Configuring for {target_platform}")
    ret = rknn.config(
        mean_values=[[123.675, 116.28, 103.53]],
        std_values=[[58.395, 57.12, 57.375]],
        target_platform=target_platform,
        quantized_dtype='asymmetric_quantized-u8' if quantize else 'float16'
    )
    if ret != 0:
        print('Config failed!')
        return False
    
    # Load ONNX model
    print('Loading model')
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print('Load model failed!')
        return False
    
    # Build model
    print('Building model')
    ret = rknn.build(do_quantization=quantize)
    if ret != 0:
        print('Build model failed!')
        return False
    
    # Export RKNN model
    print('Export RKNN model')
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print('Export failed!')
        return False
    
    # Release
    rknn.release()
    print(f"Successfully converted to {rknn_path}")
    return True

def main():
    models_path = Path(os.environ.get('MODELS_PATH', '.'))
    ddrnet_onnx = models_path / 'onnx' / 'ddrnet23-slim.onnx'
    
    if not ddrnet_onnx.exists():
        print(f"DDRNet ONNX model not found at {ddrnet_onnx}")
        return
    
    print("Converting DDRNet models...")
    
    # NPU version (quantized)
    convert_ddrnet_to_rknn(
        str(ddrnet_onnx),
        str(models_path / 'rknn' / 'ddrnet_npu.rknn'),
        target_platform='rk3588',
        quantize=True
    )
    
    # GPU version (FP16)
    convert_ddrnet_to_rknn(
        str(ddrnet_onnx),
        str(models_path / 'rknn' / 'ddrnet_gpu.rknn'),
        target_platform='rk3588',
        quantize=False
    )
    
    # CPU version (quantized)
    convert_ddrnet_to_rknn(
        str(ddrnet_onnx),
        str(models_path / 'rknn' / 'ddrnet_cpu.rknn'),
        target_platform='rk3588',
        quantize=True
    )

if __name__ == "__main__":
    main()
EOF
    
    # Run RKNN conversion
    MODELS_PATH="$MODELS_PATH" python3 "$RESULTS_DIR/convert_ddrnet_to_rknn.py"
    
    success "RKNN model conversion completed"
    
    cat > "$RESULTS_DIR/quantize_model.py" << 'EOF'
#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

import openvino as ov
import nncf

def preprocess_image(image_path, target_size=(1024, 2048)):
    """Preprocess image for DDRNet inference."""
    image = Image.open(image_path).convert('RGB')
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array
    image = np.array(image).astype(np.float32)
    
    # Normalize to [0, 1]
    image = image / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    
    # Convert HWC to CHW
    image = np.transpose(image, (2, 0, 1))
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def transform_fn(image_path):
    """Transformation function for NNCF calibration."""
    image = preprocess_image(image_path)
    return {"input": image}

def quantize_model(fp16_model_path, calibration_dir, output_dir):
    """Quantize FP16 model to INT8 using NNCF."""
    
    # Load FP16 model
    core = ov.Core()
    model = core.read_model(fp16_model_path + ".xml")
    
    # Get calibration image paths
    calibration_images_dir = Path(calibration_dir) / 'images'
    calibration_data = list(calibration_images_dir.glob('*.png'))[:500]
    
    if not calibration_data:
        raise ValueError("No calibration images found")
    
    print(f"Using {len(calibration_data)} images for calibration")
    
    # Create NNCF dataset
    calibration_dataset = nncf.Dataset(calibration_data, transform_fn)
    
    # Quantize model
    print("Quantizing model to INT8...")
    quantized_model = nncf.quantize(model, calibration_dataset)
    
    # Save quantized model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_name = output_path.name
    output_xml = output_path.with_suffix('.xml')
    
    ov.save_model(quantized_model, str(output_xml))
    
    print(f"Quantized model saved to {output_xml}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python quantize_model.py <fp16_model_path> <calibration_dir> <output_dir>")
        sys.exit(1)
    
    fp16_model_path = sys.argv[1]
    calibration_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    quantize_model(fp16_model_path, calibration_dir, output_dir)
EOF
    
    python3 "$RESULTS_DIR/quantize_model.py" \
        "$DDRNET_FP16_IR" \
        "$RESULTS_DIR/calibration" \
        "$DDRNET_INT8_IR"
    
    success "INT8 model quantization completed"
}

# Create benchmark script
create_benchmark_script() {
    log "Creating semantic segmentation benchmark script..."
    
    cat > "$RESULTS_DIR/benchmark_segmentation.py" << 'EOF'
#!/usr/bin/env python3

import os
import time
import numpy as np
from PIL import Image
import openvino as ov
from pathlib import Path
import json
import cv2

class OpenVINOInference:
    def __init__(self, model_path, device='GPU'):
        self.model_path = model_path
        self.device = device
        self.core = ov.Core()
        self.compiled_model = None
        self.input_layer = None
        self.output_layer = None
        
        self._load_model()
    
    def _load_model(self):
        """Load OpenVINO model."""
        model = self.core.read_model(self.model_path)
        
        # Get available devices
        available_devices = self.core.available_devices
        print(f"Available devices: {available_devices}")
        
        # Use CPU if GPU not available
        if self.device == 'GPU' and 'GPU' not in available_devices:
            print("GPU not available, falling back to CPU")
            self.device = 'CPU'
        
        self.compiled_model = self.core.compile_model(model, device_name=self.device)
        
        # Get input and output layers
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        print(f"Model loaded on device: {self.device}")
        print(f"Input shape: {self.input_layer.shape}")
        print(f"Output shape: {self.output_layer.shape}")
    
    def preprocess_image(self, image_path, target_size=(1024, 2048)):
        """Preprocess image for DDRNet inference."""
        image = Image.open(image_path).convert('RGB')
        
        # Resize image
        image = image.resize(target_size)
        
        # Convert to numpy array
        image = np.array(image).astype(np.float32)
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # Convert HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def infer(self, input_data):
        """Run inference on input data."""
        result = self.compiled_model([input_data])
        return result[self.output_layer]

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

def benchmark_segmentation(model_path, cityscapes_path, device='GPU', num_iterations=1000, warmup_iterations=100):
    """Benchmark DDRNet semantic segmentation."""
    
    # Initialize OpenVINO inference
    print("Initializing OpenVINO inference engine...")
    inference_engine = OpenVINOInference(model_path, device)
    
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
            
            # Process prediction
            pred_mask = output.squeeze()
            if len(pred_mask.shape) > 2:
                pred_mask = np.argmax(pred_mask, axis=0)
            
            # Resize to match ground truth if needed
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(pred_mask.astype(np.uint8), 
                                     (gt_mask.shape[1], gt_mask.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
            
            # Calculate IoU
            iou = calculate_miou(pred_mask, gt_mask)
            ious.append(iou)
    
    return np.mean(ious) if ious else None

def analyze_results(latencies, predictions, cityscapes_path, device):
    """Analyze and save benchmark results."""
    latencies_array = np.array(latencies)
    
    results = {
        'device': device,
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
    results_file = f"segmentation_results_{device.lower()}_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Semantic Segmentation Benchmark Results - {device}")
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
    import argparse
    
    parser = argparse.ArgumentParser(description='DDRNet Segmentation Benchmark')
    parser.add_argument('--model', required=True, help='Path to OpenVINO IR model (.xml)')
    parser.add_argument('--dataset', required=True, help='Path to Cityscapes dataset')
    parser.add_argument('--device', default='GPU', choices=['GPU', 'CPU'], help='OpenVINO device')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of benchmark iterations')
    parser.add_argument('--warmup', type=int, default=100, help='Number of warmup iterations')
    
    args = parser.parse_args()
    
    # Run benchmark
    latencies, predictions = benchmark_segmentation(
        args.model,
        args.dataset,
        args.device,
        args.iterations,
        args.warmup
    )
    
    # Analyze results
    analyze_results(latencies, predictions, args.dataset, args.device)

if __name__ == "__main__":
    main()
EOF
    
    chmod +x "$RESULTS_DIR/benchmark_segmentation.py"
}

# Run NPU benchmark
run_npu_benchmark() {
    log "Running NPU benchmark..."
    
    cd "$RESULTS_DIR"
    
    python3 benchmark_segmentation.py \
        --model "$DDRNET_RKNN_NPU" \
        --dataset "$CITYSCAPES_DATASET_PATH" \
        --device "NPU" \
        --iterations "$NUM_ITERATIONS" \
        --warmup "$WARMUP_ITERATIONS" > "$RESULTS_DIR/logs/npu_benchmark.log" 2>&1
    
    success "NPU benchmark completed"
}

# Run GPU benchmark
run_gpu_benchmark() {
    log "Running Mali GPU benchmark..."
    
    cd "$RESULTS_DIR"
    
    python3 benchmark_segmentation.py \
        --model "$DDRNET_RKNN_GPU" \
        --dataset "$CITYSCAPES_DATASET_PATH" \
        --device "GPU" \
        --iterations "$NUM_ITERATIONS" \
        --warmup "$WARMUP_ITERATIONS" > "$RESULTS_DIR/logs/gpu_benchmark.log" 2>&1
    
    success "Mali GPU benchmark completed"
}

# Run CPU benchmark
run_cpu_benchmark() {
    log "Running ARM CPU benchmark..."
    
    cd "$RESULTS_DIR"
    
    python3 benchmark_segmentation.py \
        --model "$DDRNET_RKNN_CPU" \
        --dataset "$CITYSCAPES_DATASET_PATH" \
        --device "CPU" \
        --iterations "$NUM_ITERATIONS" \
        --warmup "$WARMUP_ITERATIONS" > "$RESULTS_DIR/logs/cpu_benchmark.log" 2>&1
    
    success "ARM CPU benchmark completed"
}

# Main execution
main() {
    log "Starting Semantic Segmentation benchmark for Radxa CM5 (RK3588S)..."
    
    check_prerequisites
    setup_results_dir
    download_model
    create_calibration_dataset
    convert_model_to_rknn
    create_benchmark_script
    
    # Run benchmarks on NPU, GPU, and CPU
    # NPU benchmark (if available)
    if [ -f "/sys/kernel/debug/rknpu/version" ]; then
        run_npu_benchmark
    else
        warning "NPU not detected, skipping NPU benchmark"
    fi
    
    # GPU benchmark (if available)
    if [ -f "/sys/class/misc/mali0/device/uevent" ]; then
        run_gpu_benchmark
    else
        warning "Mali GPU not detected, skipping GPU benchmark"
    fi
    
    # CPU benchmark (always available)
    run_cpu_benchmark
    
    success "Semantic Segmentation benchmark completed successfully!"
    
    echo ""
    echo "=================================================="
    echo "Benchmark Complete!"
    echo "=================================================="
    echo ""
    echo "Results location: $RESULTS_DIR"
    echo ""
    echo "Key files:"
    if [ -f "/sys/kernel/debug/rknpu/version" ]; then
        echo "- segmentation_results_npu_*.json: NPU benchmark results"
    fi
    if [ -f "/sys/class/misc/mali0/device/uevent" ]; then
        echo "- segmentation_results_gpu_*.json: Mali GPU benchmark results"
    fi
    echo "- segmentation_results_cpu_*.json: ARM CPU benchmark results"
    echo "- logs/: Detailed execution logs"
    echo ""
    echo "To view results:"
    echo "ls $RESULTS_DIR/*.json"
    echo ""
}

# Execute main function
main "$@"
