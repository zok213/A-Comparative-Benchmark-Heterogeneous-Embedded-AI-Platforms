#!/bin/bash

# NVIDIA Jetson Orin NX Complete Setup Script
# This script automates the installation of all required software components
# for the embedded AI benchmark suite on NVIDIA Jetson Orin NX

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

# Check if running on NVIDIA Jetson
check_jetson() {
    log "Checking if running on NVIDIA Jetson platform..."
    
    if [ ! -f /etc/nv_tegra_release ]; then
        error "This script must be run on an NVIDIA Jetson platform"
    fi
    
    # Check JetPack version
    if command -v jetson_release >/dev/null 2>&1; then
        jetson_release
    else
        cat /etc/nv_tegra_release
    fi
    
    success "NVIDIA Jetson platform detected"
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
        tree
    
    success "System packages updated"
}

# Install ORB-SLAM3 dependencies
install_orb_slam3_deps() {
    log "Installing ORB-SLAM3 dependencies..."
    
    # Install Eigen3
    sudo apt install -y libeigen3-dev
    
    # OpenCV should already be installed with JetPack
    # Check OpenCV installation
    python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')" || {
        warning "OpenCV not found, installing..."
        sudo apt install -y python3-opencv libopencv-dev
    }
    
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
    
    # Modify build script to use fewer parallel jobs (prevent OOM on Jetson)
    sed -i 's/make -j/make -j4/g' build.sh
    
    # Build ORB-SLAM3
    ./build.sh
    
    success "ORB-SLAM3 built successfully"
}

# Setup TensorRT for AI models
setup_tensorrt() {
    log "Setting up TensorRT environment..."
    
    # Check if TensorRT is available
    if [ ! -d "/usr/src/tensorrt" ]; then
        error "TensorRT not found. Please ensure JetPack 5.1.1+ is installed"
    fi
    
    # Add TensorRT to PATH
    echo 'export PATH=/usr/src/tensorrt/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/src/tensorrt/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    # Install Python TensorRT bindings
    pip3 install --upgrade pip
    pip3 install tensorrt
    
    # Install additional Python packages for AI workloads
    pip3 install \
        onnx \
        onnxruntime-gpu \
        torch \
        torchvision \
        torchaudio \
        pillow \
        scipy \
        scikit-learn \
        pandas \
        seaborn \
        tqdm \
        opencv-python
    
    success "TensorRT environment configured"
}

# Setup power monitoring tools
setup_power_monitoring() {
    log "Setting up power monitoring tools..."
    
    # Install tegrastats (should be available by default)
    if ! command -v tegrastats >/dev/null 2>&1; then
        warning "tegrastats not found - this is expected on some JetPack versions"
    fi
    
    # Install Python packages for power analysis
    pip3 install \
        pymodbus \
        pyserial \
        matplotlib \
        numpy \
        pandas
    
    # Install jtop (Jetson monitoring tool)
    pip3 install jetson-stats
    
    success "Power monitoring tools installed"
}

# Create directory structure
create_directories() {
    log "Creating project directory structure..."
    
    cd ~/
    mkdir -p benchmark_workspace/{datasets,models,results,logs,scripts}
    
    # Create subdirectories for each benchmark
    mkdir -p benchmark_workspace/datasets/{kitti,euroc,cityscapes}
    mkdir -p benchmark_workspace/models/{onnx,tensorrt}
    mkdir -p benchmark_workspace/results/{orb_slam3,3d_detection,segmentation}
    
    success "Directory structure created"
}

# Setup environment variables
setup_environment() {
    log "Setting up environment variables..."
    
    # Create environment setup script
    cat > ~/benchmark_workspace/setup_env.sh << 'EOF'
#!/bin/bash

# NVIDIA Jetson Benchmark Environment Setup

# CUDA paths (should be set by JetPack)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# TensorRT paths
export PATH=/usr/src/tensorrt/bin:$PATH
export LD_LIBRARY_PATH=/usr/src/tensorrt/lib:$LD_LIBRARY_PATH

# Benchmark workspace
export BENCHMARK_ROOT=~/benchmark_workspace
export DATASETS_ROOT=$BENCHMARK_ROOT/datasets
export MODELS_ROOT=$BENCHMARK_ROOT/models
export RESULTS_ROOT=$BENCHMARK_ROOT/results

# ORB-SLAM3 path
export ORB_SLAM3_ROOT=~/ORB_SLAM3

# Python path for custom modules
export PYTHONPATH=$BENCHMARK_ROOT/scripts:$PYTHONPATH

echo "Environment variables set for NVIDIA Jetson benchmarking"
EOF
    
    chmod +x ~/benchmark_workspace/setup_env.sh
    
    # Add to bashrc
    echo "source ~/benchmark_workspace/setup_env.sh" >> ~/.bashrc
    
    success "Environment variables configured"
}

# Optimize Jetson performance settings
optimize_jetson() {
    log "Optimizing Jetson performance settings..."
    
    # Set maximum performance mode
    if command -v nvpmodel >/dev/null 2>&1; then
        sudo nvpmodel -m 0  # Maximum performance mode
        success "Set to maximum performance mode"
    else
        warning "nvpmodel not found - performance mode not set"
    fi
    
    # Enable maximum clocks
    if command -v jetson_clocks >/dev/null 2>&1; then
        sudo jetson_clocks
        success "Maximum clocks enabled"
    else
        warning "jetson_clocks not found - clocks not optimized"
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
}

# Verify installation
verify_installation() {
    log "Verifying installation..."
    
    # Check CUDA
    if command -v nvcc >/dev/null 2>&1; then
        nvcc --version
        success "CUDA installation verified"
    else
        error "CUDA not found"
    fi
    
    # Check TensorRT
    if [ -f "/usr/src/tensorrt/bin/trtexec" ]; then
        /usr/src/tensorrt/bin/trtexec --help > /dev/null
        success "TensorRT installation verified"
    else
        error "TensorRT not found"
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
packages = ['cv2', 'numpy', 'matplotlib', 'tensorrt', 'onnx']
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
    log "Starting NVIDIA Jetson Orin NX setup for embedded AI benchmarking..."
    
    check_jetson
    update_system
    install_orb_slam3_deps
    build_orb_slam3
    setup_tensorrt
    setup_power_monitoring
    create_directories
    setup_environment
    optimize_jetson
    verify_installation
    
    success "NVIDIA Jetson Orin NX setup completed successfully!"
    
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
    echo "For detailed usage instructions, see the README.md files"
    echo "in each benchmark directory."
    echo ""
}

# Run main function
main "$@"
