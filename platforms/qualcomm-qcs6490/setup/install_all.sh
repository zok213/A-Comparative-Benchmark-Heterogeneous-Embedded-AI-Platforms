#!/bin/bash

# Qualcomm QCS6490 Complete Setup Script
# This script automates the installation of all required software components
# for the embedded AI benchmark suite on Qualcomm QCS6490

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

# Check if running on ARM64 (expected for QCS6490)
check_platform() {
    log "Checking platform architecture..."
    
    ARCH=$(uname -m)
    if [ "$ARCH" != "aarch64" ]; then
        warning "Expected ARM64 architecture, found: $ARCH"
    fi
    
    # Check for Qualcomm-specific files/directories
    if [ -d "/sys/devices/soc0" ]; then
        log "Qualcomm SoC detected"
    else
        warning "Qualcomm-specific directories not found"
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
        bc
    
    success "System packages updated"
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
    
    # Build Pangolin with limited parallelism for ARM platforms
    cmake -B build
    cmake --build build -j4
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
    
    # Modify build script to use fewer parallel jobs (prevent OOM on ARM)
    sed -i 's/make -j/make -j4/g' build.sh
    
    # Build ORB-SLAM3
    ./build.sh
    
    success "ORB-SLAM3 built successfully"
}

# Setup Qualcomm Neural Processing SDK (SNPE)
setup_snpe_sdk() {
    log "Setting up Qualcomm Neural Processing SDK (SNPE)..."
    
    # Check if SNPE is already installed
    if [ -d ~/snpe ]; then
        log "SNPE directory already exists, skipping download"
    else
        warning "SNPE SDK must be manually downloaded from Qualcomm Developer Portal"
        echo ""
        echo "Please follow these steps:"
        echo "1. Visit: https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk-ai"
        echo "2. Register and download the SNPE SDK for Linux"
        echo "3. Extract the SDK to ~/snpe/"
        echo "4. Re-run this script after SDK installation"
        echo ""
        read -p "Press Enter if you have already installed SNPE, or Ctrl+C to exit and install it first..."
    fi
    
    # Check for SNPE installation
    if [ ! -d ~/snpe ]; then
        error "SNPE SDK not found at ~/snpe/. Please install it first."
    fi
    
    # Setup SNPE environment
    SNPE_ROOT=~/snpe
    
    # Source SNPE environment setup
    if [ -f "$SNPE_ROOT/bin/envsetup.sh" ]; then
        source "$SNPE_ROOT/bin/envsetup.sh"
        success "SNPE environment sourced"
    else
        error "SNPE environment setup script not found at $SNPE_ROOT/bin/envsetup.sh"
    fi
    
    # Install Python dependencies for SNPE
    pip3 install --upgrade pip
    pip3 install \
        onnx \
        torch \
        torchvision \
        torchaudio \
        pillow \
        scipy \
        scikit-learn \
        pandas \
        seaborn \
        tqdm \
        opencv-python \
        matplotlib \
        numpy
    
    success "SNPE SDK environment configured"
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
    
    # Check for platform-specific monitoring tools
    if command -v cat /sys/class/power_supply/*/power_now >/dev/null 2>&1; then
        log "System power monitoring available"
    else
        warning "Platform-specific power monitoring tools not detected"
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
    mkdir -p benchmark_workspace/models/{onnx,dlc}
    mkdir -p benchmark_workspace/results/{orb_slam3,3d_detection,segmentation}
    
    success "Directory structure created"
}

# Setup environment variables
setup_environment() {
    log "Setting up environment variables..."
    
    # Create environment setup script
    cat > ~/benchmark_workspace/setup_env.sh << 'EOF'
#!/bin/bash

# Qualcomm QCS6490 Benchmark Environment Setup

# SNPE SDK paths
export SNPE_ROOT=~/snpe
if [ -d "$SNPE_ROOT" ]; then
    export PATH=$SNPE_ROOT/bin/x86_64-linux-clang:$PATH
    export PATH=$SNPE_ROOT/bin/aarch64-linux-gcc7.5:$PATH
    export LD_LIBRARY_PATH=$SNPE_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$SNPE_ROOT/lib/aarch64-linux-gcc7.5:$LD_LIBRARY_PATH
    export PYTHONPATH=$SNPE_ROOT/lib/python:$PYTHONPATH
fi

# Benchmark workspace
export BENCHMARK_ROOT=~/benchmark_workspace
export DATASETS_ROOT=$BENCHMARK_ROOT/datasets
export MODELS_ROOT=$BENCHMARK_ROOT/models
export RESULTS_ROOT=$BENCHMARK_ROOT/results

# ORB-SLAM3 path
export ORB_SLAM3_ROOT=~/ORB_SLAM3

# Python path for custom modules
export PYTHONPATH=$BENCHMARK_ROOT/scripts:$PYTHONPATH

echo "Environment variables set for Qualcomm QCS6490 benchmarking"
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
    
    # Increase swap if needed (for compilation)
    if [ $(free -m | awk '/^Swap:/ {print $2}') -lt 4096 ]; then
        log "Creating additional swap space..."
        sudo fallocate -l 4G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
        success "Additional swap space created"
    fi
    
    # Disable unnecessary services to free up resources
    sudo systemctl disable bluetooth || true
    sudo systemctl disable cups || true
    
    success "System performance optimized"
}

# Verify installation
verify_installation() {
    log "Verifying installation..."
    
    # Check SNPE
    if [ -d ~/snpe ]; then
        if [ -f ~/snpe/bin/aarch64-linux-gcc7.5/snpe-net-run ]; then
            success "SNPE installation verified"
        else
            warning "SNPE tools not found in expected location"
        fi
    else
        error "SNPE SDK not found"
    fi
    
    # Check ORB-SLAM3
    if [ -f "~/ORB_SLAM3/Examples/Monocular-Inertial/mono_inertial_euroc" ]; then
        success "ORB-SLAM3 build verified"
    else
        warning "ORB-SLAM3 executable not found"
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
    log "Starting Qualcomm QCS6490 setup for embedded AI benchmarking..."
    
    check_platform
    update_system
    install_orb_slam3_deps
    build_orb_slam3
    setup_snpe_sdk
    setup_power_monitoring
    create_directories
    setup_environment
    optimize_system
    verify_installation
    
    success "Qualcomm QCS6490 setup completed successfully!"
    
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
    echo "- SNPE SDK must be manually downloaded from Qualcomm Developer Portal"
    echo "- Ensure proper cooling is in place before running benchmarks"
    echo "- For power measurement, connect external power analyzer"
    echo ""
    echo "For detailed usage instructions, see the README.md files"
    echo "in each benchmark directory."
    echo ""
}

# Run main function
main "$@"
