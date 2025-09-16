#!/usr/bin/env python3

"""
KITTI Dataset Download Script with Cookie Authentication
Downloads KITTI datasets using provided authentication cookies
"""

import os
import urllib.request
import urllib.parse
import sys
from pathlib import Path

def download_with_cookies(url, filepath, cookies):
    """Download a file using cookies for authentication."""
    print(f"Downloading {filepath.name} from KITTI...")
    
    # Create cookie string
    cookie_string = "; ".join([f"{name}={value}" for name, value in cookies.items()])
    
    # Create request with cookies
    request = urllib.request.Request(url)
    request.add_header('Cookie', cookie_string)
    request.add_header('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36')
    
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
        urllib.request.urlretrieve(request, filepath, reporthook=progress_hook)
        print(f"\n✓ Successfully downloaded {filepath.name}")
        return True
    except Exception as e:
        print(f"\n✗ Failed to download {filepath.name}: {e}")
        return False

def main():
    """Download KITTI dataset files using cookies."""
    
    # KITTI authentication cookie (updated September 16, 2025 - expires 2025-10-15)
    kitti_cookies = {
        'KITTI_USER': '964868425d81a75403b6e8d110074ea42ebc0671'
    }
    
    # Dataset destination directory
    datasets_root = Path(os.environ.get('DATASETS_ROOT', '~/benchmark_workspace/datasets')).expanduser()
    kitti_dir = datasets_root / 'kitti'
    kitti_dir.mkdir(parents=True, exist_ok=True)
    
    # KITTI download URLs (these are the actual download URLs from KITTI website)
    kitti_files = {
        'data_scene_flow.zip': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip',
        'data_object_image_2.zip': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip',
        'data_object_image_3.zip': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_3.zip',
        'data_object_calib.zip': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip',
        'data_object_label_2.zip': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip'
    }
    
    print("🚗 KITTI Dataset Download with Cookies")
    print("=" * 50)
    print(f"Download directory: {kitti_dir}")
    print()
    
    success_count = 0
    total_files = len(kitti_files)
    
    for filename, url in kitti_files.items():
        filepath = kitti_dir / filename
        
        # Check if file already exists
        if filepath.exists():
            size_mb = filepath.stat().st_size // 1024 // 1024
            print(f"✓ {filename} already exists ({size_mb} MB)")
            success_count += 1
            continue
        
        # Download file
        print(f"📥 Downloading {filename}...")
        if download_with_cookies(url, filepath, kitti_cookies):
            success_count += 1
        else:
            print(f"❌ Failed to download {filename}")
            print("   This might be due to:")
            print("   1. Expired cookie")
            print("   2. Need to re-login to KITTI website")
            print("   3. Network issues")
            break
        print()
    
    print(f"\n📊 Download Summary: {success_count}/{total_files} files successful")
    
    if success_count == total_files:
        print("🎉 All KITTI files downloaded successfully!")
        print("\nNext steps:")
        print("1. Run the dataset preparation script to extract files")
        print("2. Verify dataset structure")
        return True
    else:
        print("⚠️  Some downloads failed. Please check your KITTI account and cookies.")
        print("\nTo fix this:")
        print("1. Login to https://www.cvlibs.net/datasets/kitti/")
        print("2. Extract new KITTI_USER cookie value")
        print("3. Update this script with the new cookie")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)