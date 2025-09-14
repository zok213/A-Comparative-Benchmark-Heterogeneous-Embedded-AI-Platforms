#!/bin/bash

# ORB-SLAM3 Benchmark Script for Radxa CM5 (RK3588S)
# Implements the CPU performance benchmark using EuRoC MAV dataset

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

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Configuration
EUROC_DATASET_PATH="$DATASETS_ROOT/euroc/MH01"
ORB_SLAM3_PATH="$ORB_SLAM3_ROOT"
RESULTS_DIR="$RESULTS_ROOT/orb_slam3"
LOG_FILE="$RESULTS_DIR/benchmark_$(date +%Y%m%d_%H%M%S).log"
NUM_RUNS=5

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if ORB-SLAM3 is built
    if [ ! -f "$ORB_SLAM3_PATH/Examples/Monocular-Inertial/mono_inertial_euroc" ]; then
        error "ORB-SLAM3 not found. Please run the setup script first."
    fi
    
    # Check if EuRoC dataset is available
    if [ ! -d "$EUROC_DATASET_PATH" ]; then
        error "EuRoC dataset not found at $EUROC_DATASET_PATH. Please download the dataset first."
    fi
    
    # Check if timestamps file exists
    if [ ! -f "$ORB_SLAM3_PATH/Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt" ]; then
        error "EuRoC timestamps file not found. Please check ORB-SLAM3 installation."
    fi
    
    success "Prerequisites check passed"
}

# Setup results directory
setup_results_dir() {
    log "Setting up results directory..."
    
    mkdir -p "$RESULTS_DIR"
    mkdir -p "$RESULTS_DIR/logs"
    mkdir -p "$RESULTS_DIR/trajectories"
    
    success "Results directory created: $RESULTS_DIR"
}

# Optimize RK3588S performance settings
optimize_rk3588s() {
    log "Optimizing RK3588S performance settings..."
    
    # Set CPU governor to performance mode for both clusters
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        if [ -w "$cpu" ]; then
            echo performance | sudo tee "$cpu" > /dev/null 2>&1 || true
        fi
    done
    
    # Set maximum frequencies for big cores (A76)
    for cpu in {4..7}; do
        if [ -w "/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_max_freq" ]; then
            echo 2400000 | sudo tee /sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_max_freq > /dev/null 2>&1 || true
        fi
    done
    
    # Set maximum frequencies for LITTLE cores (A55)
    for cpu in {0..3}; do
        if [ -w "/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_max_freq" ]; then
            echo 1800000 | sudo tee /sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_max_freq > /dev/null 2>&1 || true
        fi
    done
    
    # Set Mali GPU to maximum performance
    if [ -f "/sys/class/devfreq/fb000000.gpu/governor" ]; then
        echo performance | sudo tee /sys/class/devfreq/fb000000.gpu/governor > /dev/null 2>&1 || true
    fi
    
    # Set GPU to maximum frequency
    if [ -f "/sys/class/devfreq/fb000000.gpu/max_freq" ] && [ -f "/sys/class/devfreq/fb000000.gpu/min_freq" ]; then
        max_freq=$(cat /sys/class/devfreq/fb000000.gpu/max_freq 2>/dev/null || echo "")
        if [ -n "$max_freq" ]; then
            echo "$max_freq" | sudo tee /sys/class/devfreq/fb000000.gpu/min_freq > /dev/null 2>&1 || true
        fi
    fi
    
    success "RK3588S performance optimizations applied"
}

# Monitor system resources
start_monitoring() {
    log "Starting system monitoring..."
    
    # Start system monitoring
    if command -v htop >/dev/null 2>&1; then
        htop -d 1 > "$RESULTS_DIR/system_monitor_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
        HTOP_PID=$!
        echo $HTOP_PID > "$RESULTS_DIR/htop.pid"
        log "Started htop monitoring (PID: $HTOP_PID)"
    fi
    
    # Start CPU frequency monitoring
    cat > "$RESULTS_DIR/monitor_cpu.sh" << 'EOF'
#!/bin/bash
while true; do
    echo "$(date): CPU_FREQ: $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | tr '\n' ' ')"
    if [ -f /sys/class/devfreq/fb000000.gpu/cur_freq ]; then
        echo "$(date): GPU_FREQ: $(cat /sys/class/devfreq/fb000000.gpu/cur_freq 2>/dev/null)"
    fi
    if [ -f /sys/kernel/debug/rknpu/version ]; then
        echo "$(date): NPU_STATUS: $(cat /sys/kernel/debug/rknpu/version 2>/dev/null)"
    fi
    sleep 1
done
EOF
    chmod +x "$RESULTS_DIR/monitor_cpu.sh"
    "$RESULTS_DIR/monitor_cpu.sh" > "$RESULTS_DIR/freq_monitor_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    FREQ_MON_PID=$!
    echo $FREQ_MON_PID > "$RESULTS_DIR/freq_monitor.pid"
    log "Started frequency monitoring (PID: $FREQ_MON_PID)"
    
    # Start temperature monitoring
    cat > "$RESULTS_DIR/monitor_temp.sh" << 'EOF'
#!/bin/bash
while true; do
    echo "$(date): TEMP: $(sensors 2>/dev/null | grep -E '(Core|Package|temp)' | grep -E '\+[0-9]+\.[0-9]+°C' || echo 'N/A')"
    sleep 5
done
EOF
    chmod +x "$RESULTS_DIR/monitor_temp.sh"
    "$RESULTS_DIR/monitor_temp.sh" > "$RESULTS_DIR/temp_monitor_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    TEMP_MON_PID=$!
    echo $TEMP_MON_PID > "$RESULTS_DIR/temp_monitor.pid"
    log "Started temperature monitoring (PID: $TEMP_MON_PID)"
}

# Stop system monitoring
stop_monitoring() {
    log "Stopping system monitoring..."
    
    # Stop htop
    if [ -f "$RESULTS_DIR/htop.pid" ]; then
        HTOP_PID=$(cat "$RESULTS_DIR/htop.pid")
        kill $HTOP_PID 2>/dev/null || true
        rm "$RESULTS_DIR/htop.pid"
        log "Stopped htop monitoring"
    fi
    
    # Stop frequency monitoring
    if [ -f "$RESULTS_DIR/freq_monitor.pid" ]; then
        FREQ_MON_PID=$(cat "$RESULTS_DIR/freq_monitor.pid")
        kill $FREQ_MON_PID 2>/dev/null || true
        rm "$RESULTS_DIR/freq_monitor.pid"
        log "Stopped frequency monitoring"
    fi
    
    # Stop temperature monitoring
    if [ -f "$RESULTS_DIR/temp_monitor.pid" ]; then
        TEMP_MON_PID=$(cat "$RESULTS_DIR/temp_monitor.pid")
        kill $TEMP_MON_PID 2>/dev/null || true
        rm "$RESULTS_DIR/temp_monitor.pid"
        log "Stopped temperature monitoring"
    fi
}

# Ensure system is in quiescent state
ensure_quiescent_state() {
    log "Ensuring system is in quiescent state..."
    
    # Wait for system to settle
    sleep 5
    
    # Check CPU usage
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}' || echo "0")
    log "Current CPU usage: ${CPU_USAGE}%"
    
    # Wait if CPU usage is too high
    local wait_count=0
    while (( $(echo "$CPU_USAGE > 20" | bc -l 2>/dev/null || echo "0") )); do
        if [ $wait_count -gt 30 ]; then
            warning "CPU usage still high after waiting, proceeding anyway"
            break
        fi
        log "CPU usage too high (${CPU_USAGE}%), waiting..."
        sleep 10
        CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}' || echo "0")
        ((wait_count++))
    done
    
    # Check thermal state
    if command -v sensors >/dev/null 2>&1; then
        TEMP=$(sensors 2>/dev/null | grep -E 'Core.*\+[0-9]+' | head -1 | grep -oE '\+[0-9]+\.[0-9]+' | tr -d '+' || echo "0")
        if [ -n "$TEMP" ] && (( $(echo "$TEMP > 80" | bc -l 2>/dev/null || echo "0") )); then
            warning "CPU temperature is high (${TEMP}°C), consider improving cooling"
        fi
    fi
    
    success "System is in quiescent state"
}

# Run single ORB-SLAM3 benchmark
run_single_benchmark() {
    local run_number=$1
    local run_log="$RESULTS_DIR/logs/run_${run_number}_$(date +%Y%m%d_%H%M%S).log"
    
    log "Running ORB-SLAM3 benchmark (Run $run_number/$NUM_RUNS)..."
    
    # Ensure quiescent state before each run
    ensure_quiescent_state
    
    # Change to ORB-SLAM3 directory
    cd "$ORB_SLAM3_PATH"
    
    # Run ORB-SLAM3 with timing
    local start_time=$(date +%s.%3N)
    
    # Use taskset to bind to specific CPU cores for consistency (RK3588S has 8 cores, use big cores for performance)
    timeout 300 taskset -c 4-7 ./Examples/Monocular-Inertial/mono_inertial_euroc \
        ./Vocabulary/ORBvoc.txt \
        ./Examples/Monocular-Inertial/EuRoC.yaml \
        "$EUROC_DATASET_PATH" \
        ./Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt \
        > "$run_log" 2>&1 || {
            log "ORB-SLAM3 run $run_number failed or timed out"
            return 1
        }
    
    local end_time=$(date +%s.%3N)
    local duration=$(echo "$end_time - $start_time" | bc)
    
    # Extract performance metrics from log
    local total_frames=$(grep -c "Frame processing time" "$run_log" || echo "0")
    
    if [ "$total_frames" -gt 0 ]; then
        # Calculate metrics
        echo "Run $run_number completed successfully" >> "$RESULTS_DIR/summary.txt"
        echo "Duration: ${duration}s" >> "$RESULTS_DIR/summary.txt"
        echo "Total frames: $total_frames" >> "$RESULTS_DIR/summary.txt"
        echo "Average FPS: $(echo "scale=2; $total_frames / $duration" | bc)" >> "$RESULTS_DIR/summary.txt"
        echo "---" >> "$RESULTS_DIR/summary.txt"
        
        success "Run $run_number completed: $total_frames frames in ${duration}s"
    else
        log "Run $run_number failed: No frames processed"
        return 1
    fi
}

# Run multiple benchmark iterations
run_benchmark_suite() {
    log "Starting ORB-SLAM3 benchmark suite ($NUM_RUNS runs)..."
    
    # Initialize summary file
    echo "ORB-SLAM3 Benchmark Results - Radxa CM5 (RK3588S)" > "$RESULTS_DIR/summary.txt"
    echo "Date: $(date)" >> "$RESULTS_DIR/summary.txt"
    echo "Dataset: EuRoC MAV MH01" >> "$RESULTS_DIR/summary.txt"
    echo "Number of runs: $NUM_RUNS" >> "$RESULTS_DIR/summary.txt"
    echo "CPU: RK3588S (8-core ARM: 4x A76 @ 2.4GHz + 4x A55 @ 1.8GHz)" >> "$RESULTS_DIR/summary.txt"
    echo "=================================" >> "$RESULTS_DIR/summary.txt"
    
    local successful_runs=0
    
    for i in $(seq 1 $NUM_RUNS); do
        if run_single_benchmark $i; then
            ((successful_runs++))
        fi
        
        # Wait between runs
        if [ $i -lt $NUM_RUNS ]; then
            log "Waiting 30 seconds before next run..."
            sleep 30
        fi
    done
    
    log "Benchmark suite completed: $successful_runs/$NUM_RUNS successful runs"
}

# Analyze results
analyze_results() {
    log "Analyzing benchmark results..."
    
    # Create Python analysis script
    cat > "$RESULTS_DIR/analyze_results.py" << 'EOF'
#!/usr/bin/env python3

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_log_file(log_file_path):
    """Parse ORB-SLAM3 log file to extract frame processing times."""
    processing_times_ms = []
    time_regex = re.compile(r"Frame processing time: (\d+\.\d+)\s+ms")
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                match = time_regex.search(line)
                if match:
                    time_ms = float(match.group(1))
                    processing_times_ms.append(time_ms)
    except FileNotFoundError:
        print(f"Error: Log file not found at '{log_file_path}'")
        return []
    
    return processing_times_ms

def calculate_metrics(processing_times_ms):
    """Calculate performance metrics from processing times."""
    if not processing_times_ms:
        return {}
    
    times_array = np.array(processing_times_ms)
    
    total_frames = len(times_array)
    total_duration_s = np.sum(times_array) / 1000.0
    throughput_fps = total_frames / total_duration_s if total_duration_s > 0 else 0.0
    p99_latency_ms = np.percentile(times_array, 99)
    mean_latency_ms = np.mean(times_array)
    std_dev_latency_ms = np.std(times_array)
    
    return {
        "total_frames": total_frames,
        "total_duration_s": total_duration_s,
        "throughput_fps": throughput_fps,
        "p99_latency_ms": p99_latency_ms,
        "mean_latency_ms": mean_latency_ms,
        "std_dev_latency_ms": std_dev_latency_ms
    }

def main():
    results_dir = Path(os.environ.get('RESULTS_DIR', '.'))
    logs_dir = results_dir / 'logs'
    
    all_metrics = []
    
    # Process all log files
    for log_file in logs_dir.glob('run_*.log'):
        print(f"Processing {log_file.name}...")
        times = parse_log_file(log_file)
        if times:
            metrics = calculate_metrics(times)
            metrics['log_file'] = log_file.name
            all_metrics.append(metrics)
    
    if not all_metrics:
        print("No valid log files found!")
        return
    
    # Calculate aggregate statistics
    throughputs = [m['throughput_fps'] for m in all_metrics]
    p99_latencies = [m['p99_latency_ms'] for m in all_metrics]
    mean_latencies = [m['mean_latency_ms'] for m in all_metrics]
    
    print("\n" + "="*50)
    print("ORB-SLAM3 Performance Analysis - Radxa CM5 (RK3588S)")
    print("="*50)
    print(f"Number of runs: {len(all_metrics)}")
    print(f"Average throughput: {np.mean(throughputs):.2f} ± {np.std(throughputs):.2f} FPS")
    print(f"Average P99 latency: {np.mean(p99_latencies):.2f} ± {np.std(p99_latencies):.2f} ms")
    print(f"Average mean latency: {np.mean(mean_latencies):.2f} ± {np.std(mean_latencies):.2f} ms")
    
    # Save detailed results
    with open(results_dir / 'detailed_analysis.txt', 'w') as f:
        f.write("ORB-SLAM3 Detailed Performance Analysis - Radxa CM5 (RK3588S)\n")
        f.write("="*65 + "\n\n")
        
        for i, metrics in enumerate(all_metrics, 1):
            f.write(f"Run {i} ({metrics['log_file']}):\n")
            f.write(f"  Total frames: {metrics['total_frames']}\n")
            f.write(f"  Duration: {metrics['total_duration_s']:.2f}s\n")
            f.write(f"  Throughput: {metrics['throughput_fps']:.2f} FPS\n")
            f.write(f"  P99 latency: {metrics['p99_latency_ms']:.2f} ms\n")
            f.write(f"  Mean latency: {metrics['mean_latency_ms']:.2f} ms\n")
            f.write(f"  Std dev latency: {metrics['std_dev_latency_ms']:.2f} ms\n\n")
        
        f.write("Summary Statistics:\n")
        f.write(f"Average throughput: {np.mean(throughputs):.2f} ± {np.std(throughputs):.2f} FPS\n")
        f.write(f"Average P99 latency: {np.mean(p99_latencies):.2f} ± {np.std(p99_latencies):.2f} ms\n")
        f.write(f"Average mean latency: {np.mean(mean_latencies):.2f} ± {np.std(mean_latencies):.2f} ms\n")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Throughput plot
    ax1.bar(range(1, len(throughputs) + 1), throughputs)
    ax1.set_xlabel('Run Number')
    ax1.set_ylabel('Throughput (FPS)')
    ax1.set_title('ORB-SLAM3 Throughput per Run (RK3588S)')
    ax1.grid(True, alpha=0.3)
    
    # Latency plot
    ax2.bar(range(1, len(p99_latencies) + 1), p99_latencies)
    ax2.set_xlabel('Run Number')
    ax2.set_ylabel('P99 Latency (ms)')
    ax2.set_title('ORB-SLAM3 P99 Latency per Run (RK3588S)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'performance_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nResults saved to {results_dir}")
    print(f"Visualization saved as performance_analysis.png")

if __name__ == "__main__":
    main()
EOF
    
    # Run analysis
    cd "$RESULTS_DIR"
    RESULTS_DIR="$RESULTS_DIR" python3 analyze_results.py
    
    success "Results analysis completed"
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    stop_monitoring
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    log "Starting ORB-SLAM3 benchmark for Radxa CM5 (RK3588S)..."
    
    check_prerequisites
    setup_results_dir
    optimize_rk3588s
    start_monitoring
    run_benchmark_suite
    analyze_results
    
    success "ORB-SLAM3 benchmark completed successfully!"
    
    echo ""
    echo "=================================================="
    echo "Benchmark Complete!"
    echo "=================================================="
    echo ""
    echo "Results location: $RESULTS_DIR"
    echo ""
    echo "Key files:"
    echo "- summary.txt: Basic run summary"
    echo "- detailed_analysis.txt: Detailed performance metrics"
    echo "- performance_analysis.png: Performance visualization"
    echo "- logs/: Individual run logs"
    echo "- *_monitor_*.log: System monitoring data"
    echo ""
    echo "To view results:"
    echo "cat $RESULTS_DIR/detailed_analysis.txt"
    echo ""
}

# Execute main function
main "$@"
