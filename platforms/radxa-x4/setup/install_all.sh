#!/bin/bash

# Radxa X4 (Intel N100) Complete Setup Script
# This script automates the installation of all required software components
# for the embedded AI benchmark suite on Radxa X4

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

# Check if running on x86_64 (expected for Intel N100)
check_platform() {
    log "Checking platform architecture..."
    
    ARCH=$(uname -m)
    if [ "$ARCH" != "x86_64" ]; then
        error "Expected x86_64 architecture, found: $ARCH"
    fi
    
    # Check for Intel CPU
    if grep -q "Intel" /proc/cpuinfo; then
        CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
        log "Intel CPU detected: $CPU_MODEL"
    else
        warning "Intel CPU not detected"
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
        ocl-icd-libopencl1 \
        opencl-headers \
        clinfo
    
    success "System packages updated"
}

# Install Intel GPU drivers and compute runtime
install_intel_gpu_drivers() {
    log "Installing Intel GPU drivers and compute runtime..."
    
    # Add Intel GPU repository
    wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | sudo apt-key add -
    echo "deb [arch=amd64] https://repositories.intel.com/graphics/ubuntu focal main" | sudo tee /etc/apt/sources.list.d/intel-graphics.list
    
    sudo apt update
    
    # Install Intel GPU drivers
    sudo apt install -y \
        intel-media-va-driver-non-free \
        intel-level-zero-gpu \
        level-zero \
        intel-opencl-icd
    
    # Verify OpenCL installation
    if command -v clinfo >/dev/null 2>&1; then
        log "OpenCL devices:"
        clinfo -l || warning "No OpenCL devices found"
    fi
    
    success "Intel GPU drivers installed"
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

# Install Intel OpenVINO Toolkit
install_openvino() {
    log "Installing Intel OpenVINO Toolkit..."
    
    # Install OpenVINO via pip (recommended method)
    pip3 install --upgrade pip
    pip3 install openvino openvino-dev
    
    # Install additional OpenVINO tools
    pip3 install \
        openvino-telemetry \
        nncf \
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
    
    # Verify OpenVINO installation
    python3 -c "import openvino as ov; print(f'OpenVINO version: {ov.__version__}')" || {
        error "OpenVINO installation failed"
    }
    
    # Check available devices
    python3 -c "
import openvino as ov
core = ov.Core()
devices = core.available_devices
print('Available OpenVINO devices:', devices)
for device in devices:
    try:
        print(f'Device {device}: {core.get_property(device, \"FULL_DEVICE_NAME\")}')
    except:
        print(f'Device {device}: Properties not available')
"
    
    success "OpenVINO Toolkit installed"
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

# Radxa X4 (Intel N100) Benchmark Environment Setup

# OpenVINO environment (if installed via archive)
# Note: pip installation doesn't require explicit environment setup

# Benchmark workspace
export BENCHMARK_ROOT=~/benchmark_workspace
export DATASETS_ROOT=$BENCHMARK_ROOT/datasets
export MODELS_ROOT=$BENCHMARK_ROOT/models
export RESULTS_ROOT=$BENCHMARK_ROOT/results

# ORB-SLAM3 path
export ORB_SLAM3_ROOT=~/ORB_SLAM3

# Python path for custom modules
export PYTHONPATH=$BENCHMARK_ROOT/scripts:$PYTHONPATH

# Intel GPU environment
export NEOReadDebugKeys=1
export OverrideGpuAddressSpace=48

echo "Environment variables set for Radxa X4 (Intel N100) benchmarking"
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
    
    # Check OpenVINO
    python3 -c "
import openvino as ov
print(f'OpenVINO version: {ov.__version__}')
core = ov.Core()
devices = core.available_devices
print(f'Available devices: {devices}')
if 'GPU' in devices:
    print('✓ Intel GPU support available')
else:
    print('✗ Intel GPU support not available')
" || error "OpenVINO verification failed"
    
    # Check ORB-SLAM3
    if [ -f "~/ORB_SLAM3/Examples/Monocular-Inertial/mono_inertial_euroc" ]; then
        success "ORB-SLAM3 build verified"
    else
        warning "ORB-SLAM3 executable not found"
    fi
    
    # Check OpenCL
    if command -v clinfo >/dev/null 2>&1; then
        clinfo -l > /dev/null && success "OpenCL installation verified" || warning "OpenCL devices not found"
    fi
    
    # Check Python packages
    python3 -c "
import sys
packages = ['cv2', 'numpy', 'matplotlib', 'openvino', 'onnx', 'torch']
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
    log "Starting Radxa X4 (Intel N100) setup for embedded AI benchmarking..."
    
    check_platform
    update_system
    install_intel_gpu_drivers
    install_orb_slam3_deps
    build_orb_slam3
    install_openvino
    setup_power_monitoring
    create_directories
    setup_environment
    optimize_system
    verify_installation
    
    success "Radxa X4 (Intel N100) setup completed successfully!"
    
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
    echo "- Intel GPU acceleration requires proper driver installation"
    echo ""
    echo "For detailed usage instructions, see the README.md files"
    echo "in each benchmark directory."
    echo ""
}

# Run main function
main "$@"
