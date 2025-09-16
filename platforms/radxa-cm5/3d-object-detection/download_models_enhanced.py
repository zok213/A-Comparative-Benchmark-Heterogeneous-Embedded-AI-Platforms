#!/usr/bin/env python3
"""
Enhanced Model Download Script for 3D Object Detection Benchmark
Radxa CM5 (RK3588S) Platform - September 16, 2025

Downloads CREStereo and PointPillars ONNX models with proper error handling,
progress tracking, and verification.
"""

import os
import urllib.request
import sys
import hashlib
from pathlib import Path
import ssl

def create_ssl_context():
    """Create SSL context for downloads."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

def download_file_with_progress(url, filepath, expected_size=None):
    """Download a file with detailed progress indication and verification."""
    print(f"🔄 Downloading {filepath.name}...")
    print(f"📍 Source: {url}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            mb_downloaded = downloaded // 1024 // 1024
            mb_total = total_size // 1024 // 1024
            sys.stdout.write(f"\r📥 Progress: {percent}% ({mb_downloaded}/{mb_total} MB)")
            sys.stdout.flush()
        else:
            mb_downloaded = downloaded // 1024 // 1024
            sys.stdout.write(f"\r📥 Downloaded: {mb_downloaded} MB")
            sys.stdout.flush()
    
    try:
        # Create SSL context for HTTPS downloads
        ssl_context = create_ssl_context()
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
        print(f"\n✅ Successfully downloaded {filepath.name}")
        
        # Verify file size
        actual_size = filepath.stat().st_size
        print(f"📊 File size: {actual_size // 1024 // 1024} MB")
        
        if expected_size and abs(actual_size - expected_size) > expected_size * 0.1:
            print(f"⚠️  Warning: File size differs from expected ({expected_size // 1024 // 1024} MB)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download {filepath.name}: {e}")
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
            print(f"⚠️  Warning: {filepath.name} seems too small ({size} bytes)")
            return False
        
        # Check ONNX magic bytes (optional, basic check)
        with open(filepath, 'rb') as f:
            header = f.read(8)
            # ONNX files typically start with protobuf header
            if not header:
                print(f"⚠️  Warning: {filepath.name} appears to be empty")
                return False
        
        print(f"✅ {filepath.name} appears to be a valid ONNX model")
        return True
        
    except Exception as e:
        print(f"⚠️  Could not verify {filepath.name}: {e}")
        return False

def main():
    """Download ONNX models for 3D Object Detection benchmark."""
    print("🤖 Enhanced ONNX Model Download for 3D Object Detection")
    print("=" * 60)
    
    # Setup paths
    models_dir = Path(os.environ.get('MODELS_PATH', '~/benchmark_workspace/models')).expanduser() / 'onnx'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Models directory: {models_dir}")
    
    # Model definitions with actual download URLs
    models = {
        'crestereo.onnx': {
            'url': 'https://github.com/PINTO0309/PINTO_model_zoo/raw/main/284_CREStereo/crestereo_init_iter2_480x640.onnx',
            'expected_size': 50 * 1024 * 1024,  # ~50MB
            'description': 'CREStereo stereo depth estimation model'
        },
        'pointpillars.onnx': {
            'url': 'https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars/releases/download/v1.0/pointpillars.onnx',
            'expected_size': 100 * 1024 * 1024,  # ~100MB  
            'description': 'PointPillars 3D object detection model'
        }
    }
    
    print(f"🎯 Target models: {len(models)} ONNX files")
    print()
    
    success_count = 0
    for filename, model_info in models.items():
        filepath = models_dir / filename
        
        print(f"📦 Processing: {filename}")
        print(f"📝 Description: {model_info['description']}")
        
        if filepath.exists():
            print(f"✅ {filename} already exists")
            if verify_onnx_model(filepath):
                success_count += 1
            else:
                print(f"🔄 Re-downloading due to verification failure...")
                filepath.unlink()
                if download_file_with_progress(model_info['url'], filepath, model_info['expected_size']):
                    if verify_onnx_model(filepath):
                        success_count += 1
        else:
            if download_file_with_progress(model_info['url'], filepath, model_info['expected_size']):
                if verify_onnx_model(filepath):
                    success_count += 1
        
        print("-" * 40)
    
    print(f"\n📊 Download Summary: {success_count}/{len(models)} models ready")
    
    if success_count == len(models):
        print("🎉 All ONNX models downloaded successfully!")
        print("\nNext steps:")
        print("1. Run RKNN model conversion: ./run_benchmark.sh --convert-only")
        print("2. Extract KITTI datasets if not done yet")
        print("3. Execute full benchmark: ./run_benchmark.sh")
    else:
        print("⚠️  Some models failed to download. Please check:")
        print("- Internet connectivity")
        print("- Available disk space")
        print("- File permissions")
        
        # Show what's missing
        for filename, model_info in models.items():
            filepath = models_dir / filename
            if not filepath.exists() or not verify_onnx_model(filepath):
                print(f"❌ Missing: {filename}")
                print(f"   URL: {model_info['url']}")

if __name__ == "__main__":
    main()