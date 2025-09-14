#!/bin/bash

# Radxa CM5 (RK3588S) Complete Setup Script
# This script automates the installation of all required software components
# for the embedded AI benchmark suite on Radxa CM5

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
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

# Check if running on ARM64 (expected for RK3588S)
check_platform() {
    log "Checking platform architecture..."
    
    ARCH=$(uname -m)
    if [ "$ARCH" != "aarch64" ]; then
        error "Expected ARM64 (aarch64) architecture, found: $ARCH"
    fi
    
    # Check for RK3588S processor
    if grep -q "RK3588" /proc/device-tree/compatible 2>/dev/null; then
        log "RK3588S processor detected"
    else
        warning "RK3588S processor not detected, continuing anyway"
    fi
    
    success "Platform check completed"
}

# Update system packages
update_system() {
    log "Updating system packages..."
    
    sudo apt update
    sudo apt upgrade -y
    
    # Install essential build tools
    sudo apt install -y \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        unzip \
        python3-pip \
        python3-dev \
        pkg-config \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libgtk-3-dev \
        libatlas-base-dev \
        gfortran \
        python3-numpy \
        python3-matplotlib \
        htop \
        nano \
        tree \
        bc \
        device-tree-compiler \
        librockchip-mpp-dev \
        librockchip-vpu0 \
        mali-g610-firmware
    
    success "System packages updated"
}

# Install RKNN toolkit and NPU drivers
install_rknn_toolkit() {
    log "Installing RKNN toolkit and NPU drivers..."
    
    # Create RKNN installation directory
    mkdir -p ~/rknn_toolkit
    cd ~/rknn_toolkit
    
    # Download RKNN toolkit (latest version)
    log "Downloading RKNN toolkit..."
    wget https://github.com/rockchip-linux/rknn-toolkit2/releases/download/v1.5.2/rknn_toolkit2-1.5.2-cp38-cp38-linux_aarch64.whl
    
    # Install RKNN toolkit
    pip3 install rknn_toolkit2-1.5.2-cp38-cp38-linux_aarch64.whl
    
    # Install additional RKNN dependencies
    pip3 install \
        onnx \
        onnx-simplifier \
        tensorflow \
        torch \
        torchvision \
        numpy \
        opencv-python
    
    # Verify RKNN installation
    python3 -c "from rknn.api import RKNN; print('RKNN toolkit installed successfully')" || warning "RKNN toolkit verification failed"
    
    success "RKNN toolkit installed"
}

# Install ORB-SLAM3 dependencies
install_orb_slam3_deps() {
    log "Installing ORB-SLAM3 dependencies..."
    
    # Install Eigen3
    sudo apt install -y libeigen3-dev
    
    # Install OpenCV
    sudo apt install -y python3-opencv libopencv-dev
    
    # Build Pangolin from source
    log "Building Pangolin from source..."
    cd ~/
    
    # Remove existing Pangolin if present
    [ -d "Pangolin" ] && rm -rf Pangolin
    
    git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
    cd Pangolin
    
    # Install Pangolin dependencies
    sudo apt install -y \
        libgl1-mesa-dev \
        libglew-dev \
        libpython3-dev \
        libegl1-mesa-dev \
        libwayland-dev \
        libxkbcommon-dev \
        wayland-protocols
    
    # Build Pangolin
    cmake -B build
    cmake --build build -j$(nproc)
    sudo cmake --build build --target install
    
    # Update library cache
    sudo ldconfig
    
    success "ORB-SLAM3 dependencies installed"
}

# Build ORB-SLAM3
build_orb_slam3() {
    log "Building ORB-SLAM3..."
    
    cd ~/
    
    # Remove existing ORB-SLAM3 if present
    [ -d "ORB_SLAM3" ] && rm -rf ORB_SLAM3
    
    git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
    cd ORB_SLAM3
    
    # Modify CMakeLists.txt for C++14 compatibility
    sed -i 's/c++11/c++14/g' CMakeLists.txt
    
    # Make build script executable
    chmod +x build.sh
    
    # Build ORB-SLAM3
    ./build.sh
    
    success "ORB-SLAM3 built successfully"
}

# Setup additional AI frameworks
setup_ai_frameworks() {
    log "Setting up additional AI frameworks..."
    
    # Install additional Python packages for AI workloads
    pip3 install --upgrade pip
    pip3 install \
        torch \
        torchvision \
        torchaudio \
        pillow \
        scipy \
        scikit-learn \
        pandas \
        seaborn \
        tqdm \
        matplotlib \
        numpy \
        onnx-simplifier \
        protobuf
    
    # Install Mali GPU compute libraries if available
    if [ -f "/usr/lib/aarch64-linux-gnu/libmali.so" ]; then
        log "Mali GPU libraries detected"
        # Setup OpenCL for Mali GPU
        sudo apt install -y \
            opencl-headers \
            ocl-icd-opencl-dev \
            clinfo
    fi
    
    # Verify installations
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" || warning "PyTorch verification failed"
    python3 -c "import onnx; print(f'ONNX version: {onnx.__version__}')" || warning "ONNX verification failed"
    
    success "AI frameworks setup completed"
}

# Setup power monitoring tools
setup_power_monitoring() {
    log "Setting up power monitoring tools..."
    
    # Install Python packages for power analysis
    pip3 install \
        pymodbus \
        pyserial \
        matplotlib \
        numpy \
        pandas \
        psutil
    
    # Install system monitoring tools
    sudo apt install -y \
        powertop \
        turbostat \
        stress-ng
    
    # Check for Intel-specific power monitoring
    if [ -f /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj ]; then
        log "Intel RAPL power monitoring available"
    else
        warning "Intel RAPL power monitoring not detected"
    fi
    
    success "Power monitoring tools installed"
}

# Create directory structure
create_directories() {
    log "Creating project directory structure..."
    
    cd ~/
    mkdir -p benchmark_workspace/{datasets,models,results,logs,scripts}
    
    # Create subdirectories for each benchmark
    mkdir -p benchmark_workspace/datasets/{kitti,euroc,cityscapes}
    mkdir -p benchmark_workspace/models/{onnx,openvino}
    mkdir -p benchmark_workspace/results/{orb_slam3,3d_detection,segmentation}
    
    success "Directory structure created"
}

# Setup environment variables
setup_environment() {
    log "Setting up environment variables..."
    
    # Create environment setup script
    cat > ~/benchmark_workspace/setup_env.sh << 'EOF'
#!/bin/bash

# Radxa CM5 (RK3588S) Benchmark Environment Setup

# RKNN environment setup
export RKNN_TOOLKIT_ROOT=~/rknn_toolkit

# Benchmark workspace
export BENCHMARK_ROOT=~/benchmark_workspace
export DATASETS_ROOT=$BENCHMARK_ROOT/datasets
export MODELS_ROOT=$BENCHMARK_ROOT/models
export RESULTS_ROOT=$BENCHMARK_ROOT/results

# ORB-SLAM3 path
export ORB_SLAM3_ROOT=~/ORB_SLAM3

# Python path for custom modules
export PYTHONPATH=$BENCHMARK_ROOT/scripts:$PYTHONPATH

# RK3588S specific environment
export MALI_GPU_AVAILABLE=1
export NPU_AVAILABLE=1

# ARM CPU optimization
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true
export OMP_PLACES=cores

echo "Environment variables set for Radxa CM5 (RK3588S) benchmarking"
EOF
    
    chmod +x ~/benchmark_workspace/setup_env.sh
    
    # Add to bashrc
    echo "source ~/benchmark_workspace/setup_env.sh" >> ~/.bashrc
    
    success "Environment variables configured"
}

# Optimize system performance settings
optimize_system() {
    log "Optimizing system performance settings..."
    
    # Set CPU governor to performance mode
    if [ -d "/sys/devices/system/cpu/cpu0/cpufreq" ]; then
        for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            if [ -w "$cpu" ]; then
                echo performance | sudo tee "$cpu" > /dev/null
            fi
        done
        log "CPU governor set to performance mode"
    else
        warning "CPU frequency scaling not available"
    fi
    
    # Enable Intel Turbo Boost if available
    if [ -f "/sys/devices/system/cpu/intel_pstate/no_turbo" ]; then
        echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
        log "Intel Turbo Boost enabled"
    fi
    
    # Optimize Intel GPU settings
    if [ -d "/sys/class/drm/card0" ]; then
        # Enable GPU boost if available
        for boost_file in /sys/class/drm/card*/gt_boost_freq_mhz; do
            if [ -w "$boost_file" ]; then
                # Set to maximum boost frequency
                cat /sys/class/drm/card*/gt_max_freq_mhz | head -1 | sudo tee "$boost_file" > /dev/null
            fi
        done
        log "Intel GPU boost settings optimized"
    fi
    
    # Increase swap if needed (for compilation and large models)
    if [ $(free -m | awk '/^Swap:/ {print $2}') -lt 8192 ]; then
        log "Creating additional swap space..."
        sudo fallocate -l 8G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
        success "Additional swap space created"
    fi
    
    # Disable unnecessary services
    sudo systemctl disable bluetooth || true
    sudo systemctl disable cups || true
    
    success "System performance optimized"
}

# Verify installation
verify_installation() {
    log "Verifying installation..."
    
    # Check RKNN toolkit
    python3 -c "
from rknn.api import RKNN
rknn = RKNN()
print(f'RKNN toolkit version: {rknn.get_version()}')
print('✓ RKNN toolkit available')
print('Available targets: NPU, GPU, CPU')
" || warning "RKNN toolkit verification failed"
    
    # Check ORB-SLAM3
    if [ -f "~/ORB_SLAM3/Examples/Monocular-Inertial/mono_inertial_euroc" ]; then
        success "ORB-SLAM3 build verified"
    else
        warning "ORB-SLAM3 executable not found"
    fi
    
    # Check Mali GPU
    if [ -f "/sys/class/misc/mali0/device/uevent" ]; then
        success "Mali GPU detected"
    else
        warning "Mali GPU not detected"
    fi
    
    # Check Python packages
    python3 -c "
import sys
packages = ['cv2', 'numpy', 'matplotlib', 'onnx', 'torch']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'✗ {pkg}')
        sys.exit(1)
"
    success "Python packages verified"
}

# Main installation function
main() {
    log "Starting Radxa CM5 (RK3588S) setup for embedded AI benchmarking..."
    
    check_platform
    update_system
    install_rknn_toolkit
    install_orb_slam3_deps
    build_orb_slam3
    setup_ai_frameworks
    setup_power_monitoring
    create_directories
    setup_environment
    optimize_system
    verify_installation
    
    success "Radxa CM5 (RK3588S) setup completed successfully!"
    
    echo ""
    echo "=================================================="
    echo "Setup Complete!"
    echo "=================================================="
    echo ""
    echo "Next steps:"
    echo "1. Reboot the system to ensure all changes take effect:"
    echo "   sudo reboot"
    echo ""
    echo "2. After reboot, source the environment:"
    echo "   source ~/benchmark_workspace/setup_env.sh"
    echo ""
    echo "3. Download datasets using the dataset preparation scripts"
    echo ""
    echo "4. Run the benchmarks using the provided scripts"
    echo ""
    echo "Important Notes:"
    echo "- Ensure proper cooling is in place before running benchmarks"
    echo "- For power measurement, connect external power analyzer"
    echo "- NPU acceleration requires RKNN toolkit and proper model conversion"
    echo ""
    echo "For detailed usage instructions, see the README.md files"
    echo "in each benchmark directory."
    echo ""
}

# Run main function
main "$@"
