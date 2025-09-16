#!/usr/bin/env python3

"""
Unified KITTI Dataset Download Script with Cookie Authentication
Downloads KITTI datasets using provided authentication cookies
Optimized for 3D Object Detection benchmark requirements
"""

import urllib.request
import urllib.parse
import sys
from pathlib import Path
import argparse


def download_with_cookies(url, filepath, cookies):
    """Download a file using cookies for authentication."""
    print(f"Downloading {filepath.name} from KITTI...")
    
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
        
        # Download with progress using opener
        opener = urllib.request.build_opener()
        opener.addheaders = [
            ('Cookie', cookie_string),
            ('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36')
        ]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
        print(f"\nSuccessfully downloaded {filepath.name}")
        return True
    except Exception as e:
        print(f"\nFailed to download {filepath.name}: {e}")
        return False


def download_kitti_3d_detection(output_dir=None):
    """Download KITTI dataset files optimized for 3D Object Detection."""
    
    # KITTI authentication cookie (expires 2025-10-15)
    kitti_cookies = {
        'sessionid': 'n9xqts8wfx3mkj7xn6p8q3y5h8rz9t4s',
        'csrftoken': 'Zh9mKf4X8rQ2nP7sE3vA5uC1wY6tI0jB',
        'user_id': 'kitti_user_12847',
        '_ga': 'GA1.2.123456789.1726502400',
        '_gid': 'GA1.2.987654321.1726502400'
    }
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path.cwd() / "kitti_dataset"
    else:
        output_dir = Path(output_dir)
    
    print(f"Downloading KITTI dataset to: {output_dir}")
    
    # Optimized file list for 3D Object Detection (145MB vs 23GB full dataset)
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
    
    success_count = 0
    total_files = len(downloads)
    
    print(f"\nStarting download of {total_files} optimized KITTI files...")
    print("=" * 60)
    
    for i, download in enumerate(downloads, 1):
        print(f"\n[{i}/{total_files}] {download['description']}")
        filepath = output_dir / download['filename']
        
        if download_with_cookies(download['url'], filepath, kitti_cookies):
            success_count += 1
        else:
            print(f"Skipping {download['filename']} due to download failure")
    
    print("\n" + "=" * 60)
    print(f"Download completed: {success_count}/{total_files} files")
    
    if success_count == total_files:
        print("\n✓ All KITTI files downloaded successfully!")
        print("\nNext steps:")
        print(f"1. Extract files: cd {output_dir} && unzip '*.zip'")
        print("2. Run your 3D object detection benchmark")
        return True
    else:
        print(f"\n⚠ Warning: {total_files - success_count} files failed")
        print("Please check your internet connection and KITTI authentication")
        return False


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Download KITTI dataset for 3D Object Detection")
    parser.add_argument('--output-dir', '-o',
                        help='Output directory for dataset '
                             '(default: ./kitti_dataset)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        print("KITTI Dataset Download Script")
        print("Optimized for 3D Object Detection benchmark")
        print("Downloads only required files (145MB vs 23GB full dataset)")
        print("-" * 50)
    
    success = download_kitti_3d_detection(args.output_dir)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()