#!/usr/bin/env python3

"""
Unified Model Download Script for Embedded AI Benchmark Suite
Downloads pre-trained ONNX models for all benchmark tasks across platforms
"""

import urllib.request
import urllib.parse
import sys
from pathlib import Path
import argparse
import ssl


def create_ssl_context():
    """Create SSL context for downloads."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def download_with_progress(url, filepath, expected_size=None):
    """Download a file with detailed progress indication and verification.
    
    Args:
        url: URL to download from
        filepath: Destination filepath
        expected_size: Expected file size in bytes (optional)
    """
    print(f"Downloading {filepath.name}...")
    print(f"Source: {url}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            mb_downloaded = downloaded // 1024 // 1024
            mb_total = total_size // 1024 // 1024
            sys.stdout.write(f"\rProgress: {percent}% "
                             f"({mb_downloaded}/{mb_total} MB)")
            sys.stdout.flush()
        else:
            mb_downloaded = downloaded // 1024 // 1024
            sys.stdout.write(f"\rDownloaded: {mb_downloaded} MB")
            sys.stdout.flush()
    
    try:
        # Create SSL context for HTTPS downloads
        ssl_context = create_ssl_context()
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ssl_context))
        urllib.request.install_opener(opener)
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Download the file directly
        urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
        print(f"\nSuccessfully downloaded {filepath.name}")
        
        # Verify file size
        actual_size = filepath.stat().st_size
        print(f"File size: {actual_size // 1024 // 1024} MB")
        
        if (expected_size and
                abs(actual_size - expected_size) > expected_size * 0.1):
            print(f"Warning: File size differs from expected "
                  f"({expected_size // 1024 // 1024} MB)")
        
        return True
        
    except Exception as e:
        print(f"\nFailed to download {filepath.name}: {e}")
        # Clean up partial download
        if filepath.exists():
            filepath.unlink()
        return False


def verify_onnx_model(filepath):
    """Basic verification that the downloaded file is a valid ONNX model."""
    try:
        # Check file size (ONNX models should be at least a few MB)
        size = filepath.stat().st_size
        if size < 1024 * 1024:  # Less than 1MB
            print(f"⚠️  Warning: {filepath.name} seems too small "
                  f"({size} bytes)")
            return False
        
        # Check ONNX magic bytes (optional, basic check)
        with open(filepath, 'rb') as f:
            header = f.read(8)
            if not header:
                print(f"⚠️  Warning: {filepath.name} appears to be empty")
                return False
        
        print(f"✅ {filepath.name} appears to be a valid ONNX model")
        return True
        
    except Exception as e:
        print(f"⚠️  Could not verify {filepath.name}: {e}")
        return False


def download_3d_detection_models(output_dir=None):
    """Download ONNX models for 3D Object Detection benchmark."""
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path.cwd() / "models" / "onnx"
    else:
        output_dir = Path(output_dir) / "onnx"
    
    print(f"Models directory: {output_dir}")
    
    # Model definitions for 3D Object Detection
    # Models are now hosted directly in our GitHub repository for faster downloads
    models = {
        'crestereo.onnx': {
            'url': ('https://github.com/zok213/A-Comparative-Benchmark-Heterogeneous-Embedded-AI-Platforms/'
                    'raw/main/models/crestereo.onnx'),
            'expected_size': 25 * 1024 * 1024,  # ~25MB
            'description': 'CREStereo stereo depth estimation (480x640)'
        },
        'pointpillars.onnx': {
            'url': ('https://github.com/zok213/A-Comparative-Benchmark-Heterogeneous-Embedded-AI-Platforms/'
                    'raw/main/models/pointpillars.onnx'),
            'expected_size': 18 * 1024 * 1024,  # ~18MB
            'description': 'PointPillars 3D object detection'
        }
    }
    
    return download_models_helper(models, output_dir, "3D Object Detection")


def download_semantic_segmentation_models(output_dir=None):
    """Download ONNX models for Semantic Segmentation benchmark."""
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path.cwd() / "models" / "onnx"
    else:
        output_dir = Path(output_dir) / "onnx"
    
    print(f"📁 Models directory: {output_dir}")
    
    # Model definitions for Semantic Segmentation
    models = {
        'ddrnet23-slim.onnx': {
            'url': ('https://huggingface.co/qualcomm/DDRNet23-Slim/'
                    'resolve/main/DDRNet23-Slim.onnx'),
            'expected_size': 20 * 1024 * 1024,  # ~20MB
            'description': 'DDRNet-23-Slim semantic segmentation'
        }
    }
    
    return download_models_helper(models, output_dir, "Semantic Segmentation")


def download_orb_slam_models(output_dir=None):
    """Download vocabulary and config files for ORB-SLAM3 benchmark."""
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path.cwd() / "models" / "orb_slam"
    else:
        output_dir = Path(output_dir) / "orb_slam"
    
    print(f"📁 Models directory: {output_dir}")
    
    # Model definitions for ORB-SLAM3
    models = {
        'ORBvoc.txt': {
            'url': ('https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/'
                    'Vocabulary/ORBvoc.txt'),
            'expected_size': 50 * 1024 * 1024,  # ~50MB
            'description': 'ORB vocabulary for feature matching'
        },
        'EuRoC.yaml': {
            'url': ('https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/'
                    'Examples/Monocular/EuRoC.yaml'),
            'expected_size': 2 * 1024,  # ~2KB
            'description': 'EuRoC dataset configuration'
        }
    }
    
    return download_models_helper(models, output_dir, "ORB-SLAM3")


def download_models_helper(models, output_dir, benchmark_name):
    """Helper function to download a set of models."""
    success_count = 0
    total_files = len(models)
    
    print(f"\n🎯 Downloading {benchmark_name} models...")
    print("=" * 60)
    
    for i, (filename, model_info) in enumerate(models.items(), 1):
        print(f"\n[{i}/{total_files}] {model_info['description']}")
        filepath = output_dir / filename
        
        if filepath.exists() and verify_onnx_model(filepath):
            print(f"✅ {filename} already exists and is valid")
            success_count += 1
        else:
            if filepath.exists():
                print(f"🔄 Re-downloading due to verification failure...")
                filepath.unlink()
            
            if download_with_progress(model_info['url'], filepath, 
                                      model_info['expected_size']):
                if verify_onnx_model(filepath):
                    success_count += 1
                else:
                    print(f"❌ {filename} failed verification after download")
        
        print("-" * 40)
    
    print(f"\n📊 {benchmark_name} Download Summary: "
          f"{success_count}/{total_files} models ready")
    
    return success_count == total_files


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Download ONNX models for embedded AI benchmarks")
    parser.add_argument(
        '--benchmarks', '-b', 
        choices=['3d-detection', 'semantic-segmentation', 'orb-slam', 'all'],
        nargs='+',
        default=['all'],
        help='Benchmarks to download models for (default: all)')
    parser.add_argument(
        '--output-dir', '-o',
        help='Output directory for models (default: ./models)')
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        print("🤖 Unified Model Download Script")
        print("Downloads ONNX models for embedded AI benchmarks")
        print("Supports: 3D Object Detection, Semantic Segmentation, ORB-SLAM3")
        print("-" * 60)
    
    # Expand 'all' to specific benchmarks
    if 'all' in args.benchmarks:
        benchmarks = ['3d-detection', 'semantic-segmentation', 'orb-slam']
    else:
        benchmarks = args.benchmarks
    
    total_success = True
    
    for benchmark in benchmarks:
        if benchmark == '3d-detection':
            success = download_3d_detection_models(args.output_dir)
        elif benchmark == 'semantic-segmentation':
            success = download_semantic_segmentation_models(args.output_dir)
        elif benchmark == 'orb-slam':
            success = download_orb_slam_models(args.output_dir)
        
        total_success = total_success and success
        
        if len(benchmarks) > 1:
            print("\n" + "=" * 80 + "\n")
    
    if total_success:
        print("\n🎉 All models downloaded successfully!")
        print("\nNext steps:")
        print("1. Platform-specific model optimization (TensorRT/SNPE/RKNN)")
        print("2. Run benchmarks on your target platform")
    else:
        print("\n⚠️  Some models failed to download. Please check:")
        print("- Internet connectivity")
        print("- Available disk space") 
        print("- File permissions")
        sys.exit(1)


if __name__ == "__main__":
    main()