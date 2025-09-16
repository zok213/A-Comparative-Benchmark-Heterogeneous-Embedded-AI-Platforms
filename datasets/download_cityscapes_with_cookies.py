#!/usr/bin/env python3

"""
Unified Cityscapes Dataset Download Script with Cookie Authentication
Downloads Cityscapes datasets using provided authentication cookies
Optimized for Semantic Segmentation benchmark requirements
"""

import urllib.request
import urllib.parse
import sys
from pathlib import Path
import argparse


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
        
        # Download with progress using opener
        opener = urllib.request.build_opener()
        opener.addheaders = [
            ('Cookie', cookie_string),
            ('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36')
        ]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, filepath,
                                   reporthook=progress_hook)
        print(f"\nSuccessfully downloaded {filepath.name}")
        return True
    except Exception as e:
        print(f"\nFailed to download {filepath.name}: {e}")
        return False


def download_cityscapes_semantic_segmentation(output_dir=None):
    """Download Cityscapes dataset files for Semantic Segmentation."""
    
    # Cityscapes authentication cookie (updated September 16, 2025)
    cityscapes_cookies = {
        'PHPSESSID': 'ufq0vnvd739gnknubejjm2v1ff'
    }
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path.cwd() / "cityscapes_dataset"
    else:
        output_dir = Path(output_dir)
    
    print(f"Downloading Cityscapes dataset to: {output_dir}")
    
    # Cityscapes download URLs for semantic segmentation
    cityscapes_files = {
        'leftImg8bit_trainvaltest.zip': {
            'url': ('https://www.cityscapes-dataset.com/file-handling/'
                    '?packageID=1'),
            'description': 'RGB images for train/val/test (11GB)'
        },
        'gtFine_trainvaltest.zip': {
            'url': ('https://www.cityscapes-dataset.com/file-handling/'
                    '?packageID=2'),
            'description': 'Fine annotations for train/val/test (241MB)'
        }
    }
    
    success_count = 0
    total_files = len(cityscapes_files)
    
    print(f"\nStarting download of {total_files} Cityscapes files...")
    print("=" * 60)
    
    for i, (filename, info) in enumerate(cityscapes_files.items(), 1):
        print(f"\n[{i}/{total_files}] {info['description']}")
        filepath = output_dir / filename
        
        # Check if file already exists
        if filepath.exists():
            size_mb = filepath.stat().st_size // 1024 // 1024
            print(f"✓ {filename} already exists ({size_mb} MB)")
            success_count += 1
            continue
        
        # Download file
        if download_with_cookies(info['url'], filepath, cityscapes_cookies):
            success_count += 1
        else:
            print(f"Skipping {filename} due to download failure")
            print("   This might be due to:")
            print("   1. Expired session cookie")
            print("   2. Need to re-login to Cityscapes website")
            print("   3. Network issues")
    
    print("\n" + "=" * 60)
    print(f"Download completed: {success_count}/{total_files} files")
    
    if success_count == total_files:
        print("\n✓ All Cityscapes files downloaded successfully!")
        print("\nNext steps:")
        print(f"1. Extract files: cd {output_dir} && unzip '*.zip'")
        print("2. Run your semantic segmentation benchmark")
        return True
    else:
        print(f"\n⚠ Warning: {total_files - success_count} files failed")
        print("Please check your Cityscapes account and cookies.")
        print("\nTo fix this:")
        print("1. Login to https://www.cityscapes-dataset.com/")
        print("2. Extract new PHPSESSID cookie value")
        print("3. Update this script with the new cookie")
        return False


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Download Cityscapes dataset for Semantic Segmentation")
    parser.add_argument('--output-dir', '-o',
                        help='Output directory for dataset '
                             '(default: ./cityscapes_dataset)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Cityscapes Dataset Download Script")
        print("Optimized for Semantic Segmentation benchmark")
        print("Downloads RGB images and fine annotations")
        print("-" * 50)
    
    success = download_cityscapes_semantic_segmentation(args.output_dir)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()