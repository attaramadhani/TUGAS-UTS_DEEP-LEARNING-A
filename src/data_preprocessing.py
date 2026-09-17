"""
Modul Preprocessing Data Citra CT-Scan Paru (IQ-OTH/NCCD)
Menyediakan fungsi pemuatan citra, stratifikasi train-val-test split, dan augmentasi citra.
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

CLASS_NAMES = ['Bengin cases', 'Malignant cases', 'Normal cases']
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def load_images_from_directory(data_dir, target_size=(128, 128)):
    """
    Memuat seluruh citra CT-scan ke memori dalam bentuk numpy array.
    Normalisasi nilai piksel ke rentang [0.0, 1.0].
    """
    images = []
    labels = []
    file_paths = []
    
    print(f"[PREPROCESS] Memuat citra dari {data_dir} dengan target ukuran {target_size}...")
    for class_name in CLASS_NAMES:
        class_folder = os.path.join(data_dir, class_name)
        if not os.path.exists(class_folder):
            raise FileNotFoundError(f"Folder kelas tidak ditemukan: {class_folder}")
            
        valid_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  -> Memproses {len(valid_files)} citra kelas '{class_name}'...")
        
        for f in valid_files:
            img_path = os.path.join(class_folder, f)
            with Image.open(img_path) as img:
                img_rgb = img.convert('RGB').resize(target_size, Image.Resampling.BILINEAR)
                arr = np.array(img_rgb, dtype=np.float32) / 255.0
                images.append(arr)
                labels.append(LABEL_MAP[class_name])
                file_paths.append(img_path)
                
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"[PREPROCESS] Berhasil memuat {len(X)} citra. Bentuk X: {X.shape}, y: {y.shape}")
    return X, y, file_paths

def get_stratified_split(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Membagi dataset menjadi train, validation, dan test set dengan mempertahankan proporsi kelas (stratified).
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Total rasio harus sama dengan 1.0"
    
    # Split pertama: Pisahkan Test set
    test_size = test_ratio
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Split kedua: Pisahkan Train dan Validation set
    val_rel_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_rel_size, stratify=y_train_val, random_state=random_state
    )
    
    print(f"[SPLIT] Rasio ({int(train_ratio*100)}:{int(val_ratio*100)}:{int(test_ratio*100)}) -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def build_data_generators(X_train, y_train, batch_size=32, augment=False):
    """
    Membuat tf.data.Dataset pipeline atau generator untuk pelatihan.
    Jika augment=True, menerapkan rotasi acak, flip horizontal, pergeseran lebar/tinggi, dan zoom.
    """
    if augment:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.08,
            height_shift_range=0.08,
            zoom_range=0.08,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='nearest'
        )
        return datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
    else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        return datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
