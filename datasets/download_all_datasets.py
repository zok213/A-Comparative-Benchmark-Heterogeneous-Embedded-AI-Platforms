#!/usr/bin/env python3

"""
Unified Resource Download Script for Embedded AI Benchmark Suite
Downloads datasets and models for all benchmark tasks
Supports: KITTI, Cityscapes datasets + ONNX models for all platforms
"""

import urllib.request
import urllib.parse
import sys
from pathlib import Path
import argparse
import subprocess


def download_with_cookies(url, filepath, cookies):
    """Download a file using cookies for authentication."""
    print(f"Downloading {filepath.name}...")
    
    # Create cookie string
    cookie_string = "; ".join([f"{name}={value}"
                               for name, value in cookies.items()])
    
    # Create request with cookies
    request = urllib.request.Request(url)
    request.add_header('Cookie', cookie_string)
    request.add_header('User-Agent',
                       'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36')
    
    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, (downloaded * 100) // total_size)
            mb_downloaded = downloaded // 1024 // 1024
            mb_total = total_size // 1024 // 1024
            sys.stdout.write(f"\r{percent}% ({mb_downloaded}/{mb_total} MB)")
            sys.stdout.flush()
    
    try:
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        urllib.request.urlretrieve(request, filepath,
                                   reporthook=progress_hook)
        print(f"\n✓ Successfully downloaded {filepath.name}")
        return True
    except Exception as e:
        print(f"\n✗ Failed to download {filepath.name}: {e}")
        return False


def download_kitti_dataset(output_dir):
    """Download KITTI dataset optimized for 3D Object Detection."""
    
    # KITTI authentication cookie (expires 2025-10-15)
    kitti_cookies = {
        'sessionid': 'n9xqts8wfx3mkj7xn6p8q3y5h8rz9t4s',
        'csrftoken': 'Zh9mKf4X8rQ2nP7sE3vA5uC1wY6tI0jB',
        'user_id': 'kitti_user_12847',
        '_ga': 'GA1.2.123456789.1726502400',
        '_gid': 'GA1.2.987654321.1726502400'
    }
    
    kitti_dir = output_dir / 'kitti'
    
    # Optimized file list for 3D Object Detection (145MB vs 23GB)
    downloads = [
        {
            'url': ('https://s3.eu-central-1.amazonaws.com/avg-kitti/'
                    'data_scene_flow.zip'),
            'filename': 'data_scene_flow.zip',
            'description': 'Scene flow data for 3D object detection (142MB)'
        },
        {
            'url': ('https://s3.eu-central-1.amazonaws.com/avg-kitti/'
                    'data_object_calib.zip'),
            'filename': 'data_object_calib.zip',
            'description': 'Camera calibration data (3MB)'
        }
    ]
    
    return download_dataset_files("KITTI", kitti_dir, downloads, kitti_cookies)


def download_cityscapes_dataset(output_dir):
    """Download Cityscapes dataset for Semantic Segmentation."""
    
    # Cityscapes authentication cookie
    cityscapes_cookies = {
        'PHPSESSID': 'ufq0vnvd739gnknubejjm2v1ff'
    }
    
    cityscapes_dir = output_dir / 'cityscapes'
    
    # Required files for semantic segmentation
    downloads = [
        {
            'url': ('https://www.cityscapes-dataset.com/file-handling/'
                    '?packageID=1'),
            'filename': 'leftImg8bit_trainvaltest.zip',
            'description': 'RGB images for train/val/test (11GB)'
        },
        {
            'url': ('https://www.cityscapes-dataset.com/file-handling/'
                    '?packageID=2'),
            'filename': 'gtFine_trainvaltest.zip',
            'description': 'Fine annotations for train/val/test (241MB)'
        }
    ]
    
    return download_dataset_files("Cityscapes", cityscapes_dir, downloads,
                                  cityscapes_cookies)


def download_dataset_files(dataset_name, output_dir, downloads, cookies):
    """Download a list of files for a specific dataset."""
    
    success_count = 0
    total_files = len(downloads)
    
    print(f"\n🔽 {dataset_name} Dataset Download")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Files to download: {total_files}")
    
    for i, download in enumerate(downloads, 1):
        print(f"\n[{i}/{total_files}] {download['description']}")
        filepath = output_dir / download['filename']
        
        # Check if file already exists
        if filepath.exists():
            size_mb = filepath.stat().st_size // 1024 // 1024
            print(f"✓ {download['filename']} already exists ({size_mb} MB)")
            success_count += 1
            continue
        
        # Download file
        if download_with_cookies(download['url'], filepath, cookies):
            success_count += 1
        else:
            print(f"Skipping {download['filename']} due to download failure")
    
    print(f"\n📊 {dataset_name} Summary: {success_count}/{total_files} "
          f"files downloaded")
    return success_count == total_files


def download_models(benchmarks, output_dir):
    """Download models using the unified model downloader."""
    print(f"\n🤖 Downloading models for: {', '.join(benchmarks)}")
    
    # Call the unified model downloader
    script_dir = Path(__file__).parent
    model_script = script_dir / "download_all_models.py"
    
    cmd = [sys.executable, str(model_script), 
           "--benchmarks"] + benchmarks + ["--output-dir", str(output_dir)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Model download completed successfully")
            return True
        else:
            print(f"❌ Model download failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Failed to run model downloader: {e}")
        return False


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Download datasets and models for embedded AI benchmarks")
    parser.add_argument('--datasets', '-d', nargs='+',
                        choices=['kitti', 'cityscapes', 'all'],
                        default=['all'],
                        help='Datasets to download (default: all)')
    parser.add_argument('--models', '-m', nargs='+',
                        choices=['3d-detection', 'semantic-segmentation', 
                                'orb-slam', 'all'],
                        default=[],
                        help='Models to download (default: none, use --include-models)')
    parser.add_argument('--include-models', action='store_true',
                        help='Also download models for selected benchmarks')
    parser.add_argument('--output-dir', '-o',
                        help='Output directory for datasets and models '
                             '(default: ./)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = Path.cwd()
    
    datasets_output_dir = base_output_dir / "datasets"
    models_output_dir = base_output_dir / "models"
    
    if args.verbose:
        print("🤖 Embedded AI Benchmark Resource Downloader")
        print("=" * 60)
        print("Supported resources:")
        print("📊 Datasets:")
        print("  • KITTI: 3D Object Detection (145MB optimized)")
        print("  • Cityscapes: Semantic Segmentation (~11GB)")
        print("🧠 Models:")
        print("  • 3D Detection: CREStereo + PointPillars (ONNX)")
        print("  • Semantic Segmentation: DDRNet-23-Slim (ONNX)")
        print("  • ORB-SLAM3: Vocabulary + Config files")
        print(f"\nOutput directories:")
        print(f"  📊 Datasets: {datasets_output_dir}")
        print(f"  🧠 Models: {models_output_dir}")
        print("-" * 60)
    
    # Determine which datasets to download
    datasets_to_download = args.datasets
    if 'all' in datasets_to_download:
        datasets_to_download = ['kitti', 'cityscapes']
    
    # Determine which models to download
    models_to_download = args.models.copy()
    if args.include_models and not models_to_download:
        # Auto-determine models based on datasets
        models_to_download = []
        if 'kitti' in datasets_to_download:
            models_to_download.append('3d-detection')
        if 'cityscapes' in datasets_to_download:
            models_to_download.append('semantic-segmentation')
        if not models_to_download:
            models_to_download = ['all']
    elif 'all' in models_to_download:
        models_to_download = ['3d-detection', 'semantic-segmentation', 'orb-slam']
    
    success_results = {}
    
    # Download requested datasets
    for dataset in datasets_to_download:
        if dataset == 'kitti':
            success_results['kitti'] = download_kitti_dataset(datasets_output_dir)
        elif dataset == 'cityscapes':
            success_results['cityscapes'] = download_cityscapes_dataset(
                datasets_output_dir)
    
    # Download requested models
    if models_to_download:
        success_results['models'] = download_models(models_to_download, 
                                                    models_output_dir)
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏁 FINAL DOWNLOAD SUMMARY")
    
    total_success = 0
    total_requested = len(success_results)
    
    for resource, success in success_results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {resource.upper()}: {status}")
        if success:
            total_success += 1
    
    print(f"\nOverall: {total_success}/{total_requested} resources "
          f"downloaded successfully")
    
    if total_success == total_requested:
        print("\n🎉 All requested datasets downloaded successfully!")
        print("\nNext steps:")
        print("1. Extract downloaded ZIP files")
        print("2. Run the appropriate benchmark scripts")
        print("3. Check dataset preparation scripts in each platform")
    else:
        print("\n⚠️  Some downloads failed.")
        print("Please check your internet connection and authentication.")
        sys.exit(1)


if __name__ == "__main__":
    main()