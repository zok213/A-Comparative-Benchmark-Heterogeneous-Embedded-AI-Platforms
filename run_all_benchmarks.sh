#!/bin/bash

# Master Script to Run All Benchmarks Across All Platforms
# This script orchestrates the complete benchmark suite execution

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORMS=("nvidia-jetson" "qualcomm-qcs6490" "radxa-x4")
BENCHMARKS=("orb-slam3" "3d-object-detection" "semantic-segmentation")

# Command line argument parsing
SELECTED_PLATFORMS=()
SELECTED_BENCHMARKS=()
RUN_SETUP=false
RUN_DATASETS=false
RUN_ANALYSIS=false
POWER_LOGGING=false

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --platforms PLATFORMS    Comma-separated list of platforms to run"
    echo "                          (nvidia-jetson,qualcomm-qcs6490,radxa-x4)"
    echo "  --benchmarks BENCHMARKS  Comma-separated list of benchmarks to run"
    echo "                          (orb-slam3,3d-object-detection,semantic-segmentation)"
    echo "  --setup                  Run setup scripts before benchmarks"
    echo "  --datasets               Prepare datasets before benchmarks"
    echo "  --analysis               Run analysis after benchmarks"
    echo "  --power                  Enable power logging during benchmarks"
    echo "  --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --platforms nvidia-jetson --benchmarks orb-slam3,semantic-segmentation"
    echo "  $0 --setup --datasets --analysis --power"
    echo "  $0 --platforms nvidia-jetson,radxa-x4 --benchmarks all"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --platforms)
            IFS=',' read -ra SELECTED_PLATFORMS <<< "$2"
            shift 2
            ;;
        --benchmarks)
            if [ "$2" = "all" ]; then
                SELECTED_BENCHMARKS=("${BENCHMARKS[@]}")
            else
                IFS=',' read -ra SELECTED_BENCHMARKS <<< "$2"
            fi
            shift 2
            ;;
        --setup)
            RUN_SETUP=true
            shift
            ;;
        --datasets)
            RUN_DATASETS=true
            shift
            ;;
        --analysis)
            RUN_ANALYSIS=true
            shift
            ;;
        --power)
            POWER_LOGGING=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Set defaults if nothing specified
if [ ${#SELECTED_PLATFORMS[@]} -eq 0 ]; then
    SELECTED_PLATFORMS=("${PLATFORMS[@]}")
fi

if [ ${#SELECTED_BENCHMARKS[@]} -eq 0 ]; then
    SELECTED_BENCHMARKS=("${BENCHMARKS[@]}")
fi

# Detect current platform
detect_current_platform() {
    local current_platform=""
    
    # Check for NVIDIA Jetson
    if [ -f /etc/nv_tegra_release ]; then
        current_platform="nvidia-jetson"
    # Check for Qualcomm (ARM64 with specific characteristics)
    elif [ "$(uname -m)" = "aarch64" ] && [ -d /sys/devices/soc0 ]; then
        current_platform="qualcomm-qcs6490"
    # Check for Intel x86_64
    elif [ "$(uname -m)" = "x86_64" ] && grep -q "Intel" /proc/cpuinfo; then
        current_platform="radxa-x4"
    fi
    
    echo "$current_platform"
}

# Check if platform is supported
is_platform_selected() {
    local platform=$1
    for selected in "${SELECTED_PLATFORMS[@]}"; do
        if [ "$selected" = "$platform" ]; then
            return 0
        fi
    done
    return 1
}

# Run setup for current platform
run_platform_setup() {
    local platform=$1
    local setup_script="$SCRIPT_DIR/$platform/setup/install_all.sh"
    
    if [ -f "$setup_script" ]; then
        log "Running setup for $platform..."
        chmod +x "$setup_script"
        bash "$setup_script"
        success "Setup completed for $platform"
    else
        warning "Setup script not found for $platform: $setup_script"
    fi
}

# Prepare datasets
prepare_datasets() {
    local dataset_script="$SCRIPT_DIR/datasets/prepare_all_datasets.sh"
    
    if [ -f "$dataset_script" ]; then
        log "Preparing datasets..."
        chmod +x "$dataset_script"
        bash "$dataset_script"
        success "Dataset preparation completed"
    else
        warning "Dataset preparation script not found: $dataset_script"
    fi
}

# Start power logging
start_power_logging() {
    local platform=$1
    local benchmark=$2
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="power_${platform}_${benchmark}_${timestamp}.csv"
    
    # Check if power logging script exists
    local power_script="$SCRIPT_DIR/$platform/power-measurement/power_logger.py"
    
    if [ -f "$power_script" ] && [ "$POWER_LOGGING" = true ]; then
        log "Starting power logging for $platform/$benchmark..."
        python3 "$power_script" --output "$log_file" --duration 3600 &
        POWER_PID=$!
        echo $POWER_PID > "/tmp/power_${platform}_${benchmark}.pid"
        info "Power logging started (PID: $POWER_PID)"
        return 0
    else
        warning "Power logging not available or disabled"
        return 1
    fi
}

# Stop power logging
stop_power_logging() {
    local platform=$1
    local benchmark=$2
    local pid_file="/tmp/power_${platform}_${benchmark}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            log "Power logging stopped (PID: $pid)"
        fi
        rm -f "$pid_file"
    fi
}

# Run single benchmark
run_benchmark() {
    local platform=$1
    local benchmark=$2
    local benchmark_script="$SCRIPT_DIR/$platform/$benchmark/run_benchmark.sh"
    
    if [ ! -f "$benchmark_script" ]; then
        warning "Benchmark script not found: $benchmark_script"
        return 1
    fi
    
    log "Running $benchmark on $platform..."
    
    # Start power logging if enabled
    start_power_logging "$platform" "$benchmark"
    
    # Run benchmark
    chmod +x "$benchmark_script"
    if bash "$benchmark_script"; then
        success "$benchmark completed successfully on $platform"
        BENCHMARK_SUCCESS=true
    else
        error "$benchmark failed on $platform"
        BENCHMARK_SUCCESS=false
    fi
    
    # Stop power logging
    stop_power_logging "$platform" "$benchmark"
    
    return $BENCHMARK_SUCCESS
}

# Run all benchmarks for a platform
run_platform_benchmarks() {
    local platform=$1
    
    info "Starting benchmarks for $platform"
    
    local total_benchmarks=${#SELECTED_BENCHMARKS[@]}
    local completed_benchmarks=0
    local failed_benchmarks=0
    
    for benchmark in "${SELECTED_BENCHMARKS[@]}"; do
        echo ""
        echo "=================================================="
        echo "Running: $benchmark on $platform"
        echo "Progress: $((completed_benchmarks + 1))/$total_benchmarks"
        echo "=================================================="
        echo ""
        
        if run_benchmark "$platform" "$benchmark"; then
            ((completed_benchmarks++))
        else
            ((failed_benchmarks++))
            ((completed_benchmarks++))
        fi
        
        # Wait between benchmarks for system to cool down
        if [ $completed_benchmarks -lt $total_benchmarks ]; then
            log "Waiting 60 seconds for system cooldown..."
            sleep 60
        fi
    done
    
    info "$platform benchmarks completed: $((completed_benchmarks - failed_benchmarks))/$total_benchmarks successful"
    
    if [ $failed_benchmarks -gt 0 ]; then
        warning "$failed_benchmarks benchmark(s) failed on $platform"
    fi
}

# Run analysis
run_analysis() {
    local analysis_script="$SCRIPT_DIR/analysis/scripts/analyze_all_results.py"
    
    if [ -f "$analysis_script" ]; then
        log "Running comprehensive analysis..."
        python3 "$analysis_script" --results-root "$SCRIPT_DIR" --output-dir "$SCRIPT_DIR/analysis/results"
        success "Analysis completed"
    else
        warning "Analysis script not found: $analysis_script"
    fi
}

# Main execution
main() {
    echo ""
    echo "🚀 Embedded AI Benchmark Suite Execution"
    echo "========================================"
    echo ""
    
    # Detect current platform
    CURRENT_PLATFORM=$(detect_current_platform)
    if [ -n "$CURRENT_PLATFORM" ]; then
        info "Detected platform: $CURRENT_PLATFORM"
    else
        warning "Could not auto-detect platform"
    fi
    
    # Show configuration
    info "Selected platforms: ${SELECTED_PLATFORMS[*]}"
    info "Selected benchmarks: ${SELECTED_BENCHMARKS[*]}"
    info "Setup: $RUN_SETUP"
    info "Datasets: $RUN_DATASETS"
    info "Analysis: $RUN_ANALYSIS"
    info "Power logging: $POWER_LOGGING"
    echo ""
    
    # Verify we're running on a selected platform
    if [ -n "$CURRENT_PLATFORM" ] && ! is_platform_selected "$CURRENT_PLATFORM"; then
        error "Current platform ($CURRENT_PLATFORM) is not in selected platforms list"
    fi
    
    # Run setup if requested
    if [ "$RUN_SETUP" = true ] && [ -n "$CURRENT_PLATFORM" ]; then
        run_platform_setup "$CURRENT_PLATFORM"
        echo ""
    fi
    
    # Prepare datasets if requested
    if [ "$RUN_DATASETS" = true ]; then
        prepare_datasets
        echo ""
    fi
    
    # Run benchmarks for current platform only
    if [ -n "$CURRENT_PLATFORM" ] && is_platform_selected "$CURRENT_PLATFORM"; then
        run_platform_benchmarks "$CURRENT_PLATFORM"
    else
        warning "No valid platform detected or selected for benchmark execution"
    fi
    
    # Run analysis if requested
    if [ "$RUN_ANALYSIS" = true ]; then
        echo ""
        run_analysis
    fi
    
    # Final summary
    echo ""
    echo "🎉 Benchmark Suite Execution Complete!"
    echo "======================================"
    echo ""
    
    if [ -n "$CURRENT_PLATFORM" ]; then
        info "Results for $CURRENT_PLATFORM can be found in:"
        info "  - $SCRIPT_DIR/$CURRENT_PLATFORM/results/"
        echo ""
    fi
    
    if [ "$RUN_ANALYSIS" = true ]; then
        info "Comprehensive analysis results:"
        info "  - $SCRIPT_DIR/analysis/results/"
        echo ""
    fi
    
    info "For detailed results and visualizations, check the generated reports."
    echo ""
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    
    # Stop any running power logging
    for platform in "${PLATFORMS[@]}"; do
        for benchmark in "${BENCHMARKS[@]}"; do
            stop_power_logging "$platform" "$benchmark"
        done
    done
}

# Set trap for cleanup
trap cleanup EXIT

# Run main function
main "$@"
