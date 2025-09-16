#!/usr/bin/env python3

"""
KITTI Dataset Smart Download with Storage Management
Handles large files with streaming extraction and selective processing
"""

import urllib.request
import urllib.parse
import sys
import zipfile
import tempfile
import shutil
from pathlib import Path
import argparse


def download_and_extract_selectively(url, cookies, output_dir, max_files=None, file_pattern=None):
    """Download large ZIP and extract only selected files to save space."""
    print(f"🔄 Smart downloading with selective extraction...")
    
    # Create cookie string
    cookie_string = "; ".join([f"{name}={value}" for name, value in cookies.items()])
    
    # Create request with cookies
    request = urllib.request.Request(url)
    request.add_header('Cookie', cookie_string)
    request.add_header('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36')
    
    # Use temporary file for streaming download
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
        print(f"📥 Streaming download to temporary file...")
        
        # Progress tracking
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                mb_downloaded = (block_num * block_size) // 1024 // 1024
                mb_total = total_size // 1024 // 1024
                sys.stdout.write(f"\rProgress: {percent}% ({mb_downloaded}/{mb_total} MB)")
                sys.stdout.flush()
        
        try:
            # Download to temporary file
            urllib.request.urlretrieve(url, temp_file.name, reporthook=progress_hook)
            print(f"\n✅ Download complete, extracting selectively...")
            
            # Extract only what we need
            extracted_count = 0
            with zipfile.ZipFile(temp_file.name, 'r') as zip_file:
                file_list = zip_file.namelist()
                
                print(f"📋 Archive contains {len(file_list)} files")
                
                # Apply filters
                if file_pattern:
                    file_list = [f for f in file_list if file_pattern in f]
                    print(f"🔍 Filtered to {len(file_list)} files matching '{file_pattern}'")
                
                if max_files:
                    file_list = file_list[:max_files]
                    print(f"📊 Limited to first {max_files} files")
                
                # Extract selected files
                for file_name in file_list:
                    try:
                        zip_file.extract(file_name, output_dir)
                        extracted_count += 1
                        
                        if extracted_count % 100 == 0:
                            print(f"📁 Extracted {extracted_count} files...")
                            
                    except Exception as e:
                        print(f"⚠️ Failed to extract {file_name}: {e}")
                        
                print(f"✅ Successfully extracted {extracted_count} files")
                
        finally:
            # Clean up temporary file
            Path(temp_file.name).unlink(missing_ok=True)
            print(f"🗑️ Cleaned up temporary download file")
            
    return extracted_count > 0


def download_kitti_smart(cookies, output_dir="./kitti", mode="minimal"):
    """Download KITTI with different storage strategies."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 KITTI Smart Download Mode: {mode.upper()}")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 60)
    
    if mode == "minimal":
        # Current approach - essential files only (150MB)
        downloads = [
            {
                'url': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip',
                'filename': 'data_scene_flow.zip',
                'description': 'Scene flow data (142MB)'
            },
            {
                'url': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip', 
                'filename': 'data_object_calib.zip',
                'description': 'Camera calibration (3MB)'
            }
        ]
        
        for download in downloads:
            print(f"\n📥 {download['description']}")
            filepath = output_dir / download['filename']
            download_with_cookies(download['url'], filepath, cookies)
            
    elif mode == "sample":
        # Download subset of RGB images (1-2GB instead of 12GB)
        print(f"📊 Sample mode: Downloading subset of RGB images...")
        
        # Essential files first
        essential_url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip'
        calib_url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip'
        
        print("📥 Essential files...")
        download_with_cookies(essential_url, output_dir / 'data_scene_flow.zip', cookies)
        download_with_cookies(calib_url, output_dir / 'data_object_calib.zip', cookies)
        
        # Sample RGB images (first 1000 images instead of all ~7000)
        print("📸 RGB image sample...")
        rgb_url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip'
        download_and_extract_selectively(rgb_url, cookies, output_dir, 
                                       max_files=1000, file_pattern='.png')
        
    elif mode == "full":
        # Full download with streaming extraction
        print(f"🔄 Full mode: Complete dataset with smart extraction...")
        
        urls = [
            ('data_scene_flow.zip', 'Scene flow data'),
            ('data_object_calib.zip', 'Camera calibration'), 
            ('data_object_image_2.zip', 'RGB images (12GB - streaming)'),
            ('data_object_label_2.zip', 'Ground truth labels')
        ]
        
        for filename, desc in urls:
            print(f"\n📥 {desc}")
            url = f'https://s3.eu-central-1.amazonaws.com/avg-kitti/{filename}'
            
            if 'image_2' in filename:
                # Large file - use streaming extraction
                download_and_extract_selectively(url, cookies, output_dir)
            else:
                # Small file - normal download
                download_with_cookies(url, output_dir / filename, cookies)


def download_with_cookies(url, filepath, cookies):
    """Standard download with cookies."""
    cookie_string = "; ".join([f"{name}={value}" for name, value in cookies.items()])
    
    request = urllib.request.Request(url)
    request.add_header('Cookie', cookie_string)
    request.add_header('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36')
    
    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, (block_num * block_size * 100) // total_size)
            mb_downloaded = (block_num * block_size) // 1024 // 1024
            mb_total = total_size // 1024 // 1024
            sys.stdout.write(f"\rProgress: {percent}% ({mb_downloaded}/{mb_total} MB)")
            sys.stdout.flush()
    
    urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
    print(f"\n✅ Downloaded: {filepath.name}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KITTI Smart Download')
    parser.add_argument('--mode', choices=['minimal', 'sample', 'full'], 
                       default='minimal', help='Download mode')
    parser.add_argument('--output-dir', default='./kitti', help='Output directory')
    
    args = parser.parse_args()
    
    # You'll need to provide your KITTI cookies
    kitti_cookies = {
        # Add your KITTI session cookies here
        'session_id': 'your_session_id',
        'auth_token': 'your_auth_token'
    }
    
    download_kitti_smart(kitti_cookies, args.output_dir, args.mode)