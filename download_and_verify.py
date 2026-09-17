"""
Script Download dan Verifikasi Dataset Citra CT-Scan IQ-OTH/NCCD
Sumber: Mendeley Data (DOI: 10.17632/bhmdr45bh2.2)
Penulis Dataset: Hamdalla Alyasriy & Muayed AL-Huseiny (2020)
"""

import os
import shutil
import sys
from PIL import Image

def download_and_verify(data_dir=None):
    if data_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data', 'lung_cancer')
    
    os.makedirs(data_dir, exist_ok=True)
    classes = ['Bengin cases', 'Malignant cases', 'Normal cases']
    
    # Periksa apakah data sudah tersedia secara lokal
    already_exists = True
    for c in classes:
        p = os.path.join(data_dir, c)
        if not os.path.exists(p) or len([f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) == 0:
            already_exists = False
            break
            
    if already_exists:
        print(f"[OK] Dataset sudah tersedia di folder lokal: {data_dir}")
    else:
        print("[INFO] Mengunduh dataset IQ-OTH/NCCD dari repositori resmi via kagglehub...")
        try:
            import kagglehub
            cache_path = kagglehub.dataset_download('hamdallak/the-iqothnccd-lung-cancer-dataset')
            print(f"[OK] Dataset berhasil diunduh ke cache: {cache_path}")
            
            src_dataset_dir = os.path.join(cache_path, 'The IQ-OTHNCCD lung cancer dataset')
            if not os.path.exists(src_dataset_dir):
                src_dataset_dir = cache_path
                
            for item in os.listdir(src_dataset_dir):
                s = os.path.join(src_dataset_dir, item)
                d = os.path.join(data_dir, item)
                if os.path.isdir(s):
                    if not os.path.exists(d):
                        shutil.copytree(s, d)
                else:
                    if not os.path.exists(d):
                        shutil.copy2(s, d)
            print(f"[OK] File dataset berhasil disalin ke direktori proyek lokal: {data_dir}")
        except Exception as e:
            print(f"[ERROR] Gagal mengunduh dataset otomatis: {e}")
            sys.exit(1)
            
    # Tampilkan statistik dataset
    print("\n" + "="*60)
    print("STATISTIK DATASET CITRA CT-SCAN IQ-OTH/NCCD")
    print("="*60)
    total_images = 0
    stats = {}
    
    for c in classes:
        p = os.path.join(data_dir, c)
        if os.path.exists(p):
            files = [f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_images += len(files)
            stats[c] = len(files)
            # Sample citra
            if files:
                sample_path = os.path.join(p, files[0])
                with Image.open(sample_path) as img:
                    width, height = img.size
                    mode = img.mode
            else:
                width, height, mode = (0, 0, 'None')
            print(f"  * Kelas '{c}': {len(files)} citra | Contoh Resolusi: {width}x{height} | Mode: {mode}")
        else:
            print(f"  * Kelas '{c}': Folder tidak ditemukan!")
            
    print("-" * 60)
    print(f"  TOTAL CITRA: {total_images} citra")
    print("="*60 + "\n")
    return data_dir, stats

if __name__ == '__main__':
    download_and_verify()
