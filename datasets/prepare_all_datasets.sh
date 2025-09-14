#!/bin/bash

# Complete Dataset Preparation Script
# Downloads and prepares all required datasets for the embedded AI benchmark suite

set -e

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

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Source environment if available
if [ -f ~/benchmark_workspace/setup_env.sh ]; then
    source ~/benchmark_workspace/setup_env.sh
else
    # Set default paths
    export DATASETS_ROOT=~/benchmark_workspace/datasets
fi

# Create datasets directory
setup_datasets_dir() {
    log "Setting up datasets directory structure..."
    
    mkdir -p "$DATASETS_ROOT"/{kitti,euroc,cityscapes}
    
    success "Datasets directory structure created"
}

# Prepare EuRoC MAV dataset for ORB-SLAM3
prepare_euroc_dataset() {
    log "Preparing EuRoC MAV dataset..."
    
    local euroc_dir="$DATASETS_ROOT/euroc"
    cd "$euroc_dir"
    
    # Download Machine Hall 01 sequence
    if [ ! -d "MH01" ]; then
        log "Downloading EuRoC MAV Machine Hall 01 sequence..."
        
        wget -c http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip
        
        if [ $? -eq 0 ]; then
            log "Extracting EuRoC dataset..."
            mkdir -p MH01
            unzip MH_01_easy.zip -d MH01/
            rm MH_01_easy.zip
            success "EuRoC MAV dataset prepared"
        else
            error "Failed to download EuRoC dataset"
        fi
    else
        log "EuRoC MAV dataset already exists"
    fi
    
    # Verify dataset structure
    if [ -d "MH01/mav0" ]; then
        success "EuRoC dataset structure verified"
    else
        error "EuRoC dataset structure invalid"
    fi
}

# Prepare KITTI dataset for 3D object detection
prepare_kitti_dataset() {
    log "Preparing KITTI dataset..."
    
    local kitti_dir="$DATASETS_ROOT/kitti"
    cd "$kitti_dir"
    
    echo ""
    echo "=================================================="
    echo "KITTI Dataset Download Instructions"
    echo "=================================================="
    echo ""
    echo "The KITTI dataset requires manual registration and download."
    echo "Please follow these steps:"
    echo ""
    echo "1. Visit: https://www.cvlibs.net/datasets/kitti/"
    echo "2. Register for an account"
    echo "3. Download the following files to $kitti_dir:"
    echo "   - KITTI Stereo 2015: data_scene_flow.zip"
    echo "   - KITTI 3D Object Detection: data_object_image_2.zip"
    echo "   - KITTI 3D Object Detection: data_object_image_3.zip"
    echo "   - KITTI 3D Object Detection: data_object_calib.zip"
    echo "   - KITTI 3D Object Detection: data_object_label_2.zip"
    echo ""
    echo "4. After downloading, re-run this script to extract the files"
    echo ""
    
    # Check if files are already downloaded
    local files_to_check=(
        "data_scene_flow.zip"
        "data_object_image_2.zip"
        "data_object_image_3.zip"
        "data_object_calib.zip"
        "data_object_label_2.zip"
    )
    
    local all_files_present=true
    for file in "${files_to_check[@]}"; do
        if [ ! -f "$file" ]; then
            all_files_present=false
            break
        fi
    done
    
    if [ "$all_files_present" = true ]; then
        log "All KITTI files found, extracting..."
        
        # Create directory structure
        mkdir -p object/{training,testing}/{calib,image_2,image_3,label_2}
        
        # Extract files
        log "Extracting KITTI object detection images (left)..."
        unzip -q data_object_image_2.zip
        
        log "Extracting KITTI object detection images (right)..."
        unzip -q data_object_image_3.zip
        
        log "Extracting KITTI calibration files..."
        unzip -q data_object_calib.zip
        
        log "Extracting KITTI labels..."
        unzip -q data_object_label_2.zip
        
        # Clean up zip files
        rm -f data_object_*.zip
        
        success "KITTI dataset extracted successfully"
        
        # Verify structure
        if [ -d "object/training/image_2" ] && [ -d "object/training/image_3" ]; then
            local num_images=$(ls object/training/image_2/*.png 2>/dev/null | wc -l)
            log "Found $num_images training images"
            success "KITTI dataset structure verified"
        else
            error "KITTI dataset structure invalid"
        fi
    else
        warning "KITTI dataset files not found. Please download manually."
        return 1
    fi
}

# Prepare Cityscapes dataset for semantic segmentation
prepare_cityscapes_dataset() {
    log "Preparing Cityscapes dataset..."
    
    local cityscapes_dir="$DATASETS_ROOT/cityscapes"
    cd "$cityscapes_dir"
    
    echo ""
    echo "=================================================="
    echo "Cityscapes Dataset Download Instructions"
    echo "=================================================="
    echo ""
    echo "The Cityscapes dataset requires manual registration and download."
    echo "Please follow these steps:"
    echo ""
    echo "1. Visit: https://www.cityscapes-dataset.com/"
    echo "2. Register for an account"
    echo "3. Download the following files to $cityscapes_dir:"
    echo "   - leftImg8bit_trainvaltest.zip (11GB)"
    echo "   - gtFine_trainvaltest.zip (241MB)"
    echo ""
    echo "4. After downloading, re-run this script to extract the files"
    echo ""
    
    # Check if files are already downloaded
    if [ -f "leftImg8bit_trainvaltest.zip" ] && [ -f "gtFine_trainvaltest.zip" ]; then
        log "Cityscapes files found, extracting..."
        
        log "Extracting Cityscapes images..."
        unzip -q leftImg8bit_trainvaltest.zip
        
        log "Extracting Cityscapes ground truth..."
        unzip -q gtFine_trainvaltest.zip
        
        # Clean up zip files
        rm -f leftImg8bit_trainvaltest.zip gtFine_trainvaltest.zip
        
        success "Cityscapes dataset extracted successfully"
        
        # Verify structure
        if [ -d "leftImg8bit/val" ] && [ -d "gtFine/val" ]; then
            local num_val_images=$(find leftImg8bit/val -name "*.png" | wc -l)
            log "Found $num_val_images validation images"
            success "Cityscapes dataset structure verified"
        else
            error "Cityscapes dataset structure invalid"
        fi
    else
        warning "Cityscapes dataset files not found. Please download manually."
        return 1
    fi
}

# Create dataset summary
create_dataset_summary() {
    log "Creating dataset summary..."
    
    cat > "$DATASETS_ROOT/dataset_summary.txt" << EOF
Embedded AI Benchmark Dataset Summary
=====================================

Generated: $(date)

EuRoC MAV Dataset:
- Location: $DATASETS_ROOT/euroc/
- Purpose: ORB-SLAM3 CPU benchmark
- Sequence: Machine Hall 01 (MH01)
- Status: $([ -d "$DATASETS_ROOT/euroc/MH01" ] && echo "✓ Available" || echo "✗ Not available")

KITTI Dataset:
- Location: $DATASETS_ROOT/kitti/
- Purpose: 3D Object Detection benchmark
- Components: Stereo images, calibration, 3D labels
- Status: $([ -d "$DATASETS_ROOT/kitti/object" ] && echo "✓ Available" || echo "✗ Not available")

Cityscapes Dataset:
- Location: $DATASETS_ROOT/cityscapes/
- Purpose: Semantic Segmentation benchmark
- Components: RGB images, semantic labels
- Status: $([ -d "$DATASETS_ROOT/cityscapes/leftImg8bit" ] && echo "✓ Available" || echo "✗ Not available")

Dataset Sizes:
- EuRoC: $([ -d "$DATASETS_ROOT/euroc" ] && du -sh "$DATASETS_ROOT/euroc" 2>/dev/null | cut -f1 || echo "N/A")
- KITTI: $([ -d "$DATASETS_ROOT/kitti" ] && du -sh "$DATASETS_ROOT/kitti" 2>/dev/null | cut -f1 || echo "N/A")
- Cityscapes: $([ -d "$DATASETS_ROOT/cityscapes" ] && du -sh "$DATASETS_ROOT/cityscapes" 2>/dev/null | cut -f1 || echo "N/A")
- Total: $(du -sh "$DATASETS_ROOT" 2>/dev/null | cut -f1 || echo "N/A")

EOF
    
    success "Dataset summary created: $DATASETS_ROOT/dataset_summary.txt"
}

# Main execution
main() {
    log "Starting complete dataset preparation..."
    
    setup_datasets_dir
    
    # Prepare each dataset
    prepare_euroc_dataset
    prepare_kitti_dataset || warning "KITTI dataset preparation incomplete"
    prepare_cityscapes_dataset || warning "Cityscapes dataset preparation incomplete"
    
    create_dataset_summary
    
    success "Dataset preparation completed!"
    
    echo ""
    echo "=================================================="
    echo "Dataset Preparation Summary"
    echo "=================================================="
    echo ""
    cat "$DATASETS_ROOT/dataset_summary.txt"
    echo ""
    echo "Next steps:"
    echo "1. Verify all datasets are properly downloaded and extracted"
    echo "2. Run the benchmark scripts for each platform"
    echo ""
    echo "For manual dataset downloads, follow the instructions above"
    echo "and re-run this script after downloading."
    echo ""
}

# Execute main function
main "$@"
