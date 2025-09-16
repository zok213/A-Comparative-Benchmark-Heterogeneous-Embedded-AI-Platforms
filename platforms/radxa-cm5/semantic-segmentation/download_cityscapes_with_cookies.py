#!/usr/bin/env python3

"""
Cityscapes Dataset Download Script with Cookie Authentication
Downloads Cityscapes datasets using provided authentication cookies
"""

import os
import urllib.request
import urllib.parse
import sys
from pathlib import Path


def download_with_cookies(url, filepath, cookies):
    """Download a file using cookies for authentication."""
    print(f"Downloading {filepath.name} from Cityscapes...")
    
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


def main():
    """Download Cityscapes dataset files using cookies."""
    
    # Cityscapes authentication cookie (updated September 16, 2025)
    cityscapes_cookies = {
        'PHPSESSID': 'ufq0vnvd739gnknubejjm2v1ff'
    }
    
    # Dataset destination directory
    datasets_root = Path(os.environ.get('DATASETS_ROOT', 
                                        '~/benchmark_workspace/datasets')
                         ).expanduser()
    cityscapes_dir = datasets_root / 'cityscapes'
    cityscapes_dir.mkdir(parents=True, exist_ok=True)
    
    # Cityscapes download URLs (need to be actual download URLs)
    cityscapes_files = {
        'leftImg8bit_trainvaltest.zip': 
            'https://www.cityscapes-dataset.com/file-handling/?packageID=1',
        'gtFine_trainvaltest.zip': 
            'https://www.cityscapes-dataset.com/file-handling/?packageID=2'
    }
    
    print("🏙️  Cityscapes Dataset Download with Cookies")
    print("=" * 50)
    print(f"Download directory: {cityscapes_dir}")
    print()
    
    success_count = 0
    total_files = len(cityscapes_files)
    
    for filename, url in cityscapes_files.items():
        filepath = cityscapes_dir / filename
        
        # Check if file already exists
        if filepath.exists():
            size_mb = filepath.stat().st_size // 1024 // 1024
            print(f"✓ {filename} already exists ({size_mb} MB)")
            success_count += 1
            continue
        
        # Download file
        print(f"📥 Downloading {filename}...")
        if download_with_cookies(url, filepath, cityscapes_cookies):
            success_count += 1
        else:
            print(f"❌ Failed to download {filename}")
            print("   This might be due to:")
            print("   1. Expired session cookie")
            print("   2. Need to re-login to Cityscapes website")
            print("   3. Network issues")
            break
        print()
    
    print(f"\n📊 Download Summary: {success_count}/{total_files} "
          f"files successful")
    
    if success_count == total_files:
        print("🎉 All Cityscapes files downloaded successfully!")
        print("\nNext steps:")
        print("1. Run the dataset preparation script to extract files")
        print("2. Verify dataset structure")
        return True
    else:
        print("⚠️  Some downloads failed. Please check your Cityscapes "
              "account and cookies.")
        print("\nTo fix this:")
        print("1. Login to https://www.cityscapes-dataset.com/")
        print("2. Extract new PHPSESSID cookie value")
        print("3. Update this script with the new cookie")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)