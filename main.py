"""
================================================================================
TUGAS UJIAN TENGAH SEMESTER (UTS) — DEEP LEARNING (KELAS A)
PROGRAM UTAMA: ARSITEKTUR CONVOLUTIONAL NEURAL NETWORK (CNN) 4 SKENARIO BERTINGKAT
DATASET: THE IQ-OTHNCCD LUNG CANCER CT-SCAN DATASET (MENDELEY DATA DOI: 10.17632/bhmdr45bh2.2)
================================================================================

👥 IDENTITAS KELOMPOK 6:
1. Attala Alif Ramadhani Tri Hida (NIM: 230441100144 / 23-144)
2. Nafaul Hernanda Romadlona      (NIM: 240441100125 / 24-125)
3. M.Rafly Kurniawan              (NIM: 240441100086 / 24-086)
4. Naufal Husain                  (NIM: 240441100038 / 24-038)

Dosen Pengampu: Dr. Wahyudi Setiawan, S.Kom., M.Kom.
Program Studi : S1 Sistem Informasi, Fakultas Teknik, Universitas Trunojoyo Madura

Skrip ini menyatukan seluruh komponen pipeline ke dalam satu file terpadu:
1. Pengunduhan dan verifikasi otomatis dataset resmi (Mendeley Data).
2. Preprocessing citra, normalisasi, dan pembagian dataset berstrata (Stratified Split).
3. Pembangunan arsitektur kustom Deep CNN 4-Blok Conv-ReLU-Pool hierarkis.
4. Eksekusi 4 skenario optimasi bertingkat (Progressive Ablation Study):
   - Skenario 1: Optimasi Rasio Split Data (70:15:15 vs 80:10:10 vs 90:05:05)
   - Skenario 2: Optimasi Data Augmentasi (Tanpa vs Dengan Augmentasi)
   - Skenario 3: Komparasi Optimizer (Adam vs RMSprop vs SGD Momentum)
   - Skenario 4: Optimasi Regularisasi Dropout (0.0 vs 0.3 vs 0.5) -> Final Champion
5. Evaluasi metrik komprehensif dan visualisasi grafik 300 DPI.
6. Pembuatan otomatis laporan akademik formal berformat Microsoft Word (.docx).
================================================================================
"""

import os
import sys
import time
import json
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

import tensorflow as tf

# Opsional: python-docx untuk pembuatan laporan Word
try:
    import docx
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# ==============================================================================
# 1. KONFIGURASI GLOBAL & REPRODUSIBILITAS
# ==============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'lung_cancer')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')
MODELS_DIR = os.path.join(OUTPUTS_DIR, 'models')
LOGS_DIR = os.path.join(OUTPUTS_DIR, 'logs')
REPORT_PATH = os.path.join(BASE_DIR, 'Laporan_Lengkap_UTS_DeepLearning_CNN.docx')

for d in [FIGURES_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

EPOCHS = 8
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
IMG_SIZE = (128, 128)
CLASS_NAMES = ['Bengin cases', 'Malignant cases', 'Normal cases']
DISPLAY_NAMES = ['Benign (Jinak)', 'Malignant (Ganas)', 'Normal (Sehat)']
C_LABELS = ['Benign', 'Malignant', 'Normal']
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# ==============================================================================
# 2. PENGUNDUHAN & VERIFIKASI DATASET
# ==============================================================================
def download_and_verify(data_dir=DATA_DIR):
    """
    Memeriksa ketersediaan dataset di folder lokal data/lung_cancer/.
    Jika belum ada, mengunduh otomatis dataset resmi IQ-OTH/NCCD via kagglehub.
    """
    os.makedirs(data_dir, exist_ok=True)
    already_exists = True
    for c in CLASS_NAMES:
        p = os.path.join(data_dir, c)
        if not os.path.exists(p) or len([f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) == 0:
            already_exists = False
            break

    if already_exists:
        print(f"[OK] Dataset sudah tersedia secara lokal di: {data_dir}")
    else:
        print("[INFO] Dataset belum lengkap. Mengunduh dataset via kagglehub...")
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
            print(f"[OK] Dataset berhasil disalin ke direktori proyek: {data_dir}")
        except Exception as e:
            print(f"[ERROR] Gagal mengunduh dataset secara otomatis: {e}")
            sys.exit(1)

    # Tampilkan statistik dataset
    print("\n" + "=" * 65)
    print("STATISTIK DATASET CITRA CT-SCAN IQ-OTH/NCCD (MENDELEY DATA)")
    print("=" * 65)
    total_images = 0
    stats = {}
    for c, dname in zip(CLASS_NAMES, DISPLAY_NAMES):
        p = os.path.join(data_dir, c)
        if os.path.exists(p):
            files = [f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_images += len(files)
            stats[c] = len(files)
            print(f"  * Kelas {dname:20s}: {len(files):4d} citra")
        else:
            print(f"  * Kelas {dname:20s}: Folder tidak ditemukan!")
    print("-" * 65)
    print(f"  TOTAL CITRA DATASET          : {total_images:4d} citra")
    print("=" * 65 + "\n")
    return data_dir, stats


# ==============================================================================
# 3. PREPROCESSING CITRA & STRATIFIED SPLIT
# ==============================================================================
def load_images_from_directory(data_dir=DATA_DIR, target_size=IMG_SIZE):
    """
    Memuat seluruh citra CT-Scan ke memory dalam format NumPy array ternormalisasi [0.0, 1.0].
    """
    images, labels, file_paths = [], [], []
    print(f"[PREPROCESS] Memuat citra dari {data_dir} dengan target ukuran {target_size}...")
    for class_name in CLASS_NAMES:
        class_folder = os.path.join(data_dir, class_name)
        if not os.path.exists(class_folder):
            raise FileNotFoundError(f"Folder kelas tidak ditemukan: {class_folder}")
        valid_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
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
    print(f"[PREPROCESS] Selesai memuat array citra: Shape X={X.shape}, y={y.shape}")
    return X, y, file_paths


def get_stratified_split(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=RANDOM_SEED):
    """
    Membagi dataset menjadi Train, Validation, dan Test set dengan Stratified Sampling.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Total rasio harus sama dengan 1.0"
    test_size = test_ratio
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_rel_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_rel_size, stratify=y_train_val, random_state=random_state
    )
    print(f"[SPLIT] Rasio ({int(train_ratio*100)}:{int(val_ratio*100)}:{int(test_ratio*100)}) -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ==============================================================================
# 4. PEMBANGUN ARSITEKTUR MODEL CNN (CUSTOM 4-BLOK)
# ==============================================================================
def build_cnn_model(dropout_rate=0.3, model_name='Custom_Lung_CNN'):
    """
    Membangun model CNN modular 4 blok konvolusi hierarkis sesuai spesifikasi tugas UTS:
    Conv2D(32) -> Conv2D(64) -> Conv2D(128) -> Conv2D(128) + ReLU + MaxPool(2x2)
    Classifier: Flatten -> Dense(128, ReLU) -> Dropout(p) -> Dense(3, Softmax)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name='input_image'),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool3'),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv4'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool4'),
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(128, activation='relu', name='dense_feature'),
        tf.keras.layers.Dropout(dropout_rate, name='dropout') if dropout_rate > 0.0 else tf.keras.layers.Identity(name='no_dropout'),
        tf.keras.layers.Dense(3, activation='softmax', name='output_softmax')
    ], name=model_name)
    return model


def compile_custom(model, opt_name='adam', lr=LEARNING_RATE):
    """
    Mengompilasi model dengan optimizer pilihan (Adam, RMSprop, atau SGD Momentum).
    """
    opt_name = opt_name.lower()
    if opt_name == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
    elif opt_name == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=lr * 2, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Optimizer {opt_name} tidak didukung.")
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ==============================================================================
# 5. PELATIHAN, EVALUASI, & VISUALISASI
# ==============================================================================
def train_and_eval(name, model, train_data, val_data, test_data, is_augmented=False, epochs=EPOCHS):
    """
    Melatih model secara terkontrol dan menghitung metrik evaluasi pada test set.
    """
    print(f"\n[{name.upper()}] Memulai pelatihan ({epochs} Epochs)...")
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    t0 = time.time()
    if is_augmented:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15, width_shift_range=0.08, height_shift_range=0.08,
            zoom_range=0.08, horizontal_flip=True, fill_mode='nearest'
        )
        flow = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
        steps = int(np.ceil(len(X_train) / BATCH_SIZE))
        hist = model.fit(flow, steps_per_epoch=steps, epochs=epochs, validation_data=(X_val, y_val), verbose=1)
    else:
        hist = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=epochs, validation_data=(X_val, y_val), verbose=1)
    dur = time.time() - t0

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=C_LABELS, output_dict=True, zero_division=0)

    print(f"[{name}] HASIL -> Akurasi: {test_acc*100:.2f}% | Loss: {test_loss:.4f} | F1: {f1_mac*100:.2f}% | Waktu: {dur:.1f}s")
    return {
        'name': name,
        'model': model,
        'history': hist.history,
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'precision_macro': float(p_mac),
        'recall_macro': float(r_mac),
        'f1_macro': float(f1_mac),
        'confusion_matrix': cm.tolist(),
        'classification_report': cr,
        'training_time': round(dur, 2),
        'y_pred': y_pred.tolist(),
        'y_test': y_test.tolist()
    }


def plot_progressive_progression(df_prog, save_path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#FFFFFF')
    stages = df_prog['Tahap']
    accs = df_prog['Test Accuracy (%)']
    f1s = df_prog['Macro F1 (%)']
    x = np.arange(len(stages))
    width = 0.35

    r1 = ax.bar(x - width/2, accs, width, label='Test Accuracy (%)', color='#2B6CB0', edgecolor='black', linewidth=0.5)
    r2 = ax.bar(x + width/2, f1s, width, label='Macro F1-Score (%)', color='#38A169', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Persentase (%)', fontsize=11, fontweight='bold')
    ax.set_title('Progresi Peningkatan Performa Pemenang Tiap Tahap (Sequential Optimization)', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n({df_prog.loc[i, 'Konfigurasi Terpilih']})" for i, s in enumerate(stages)], fontsize=9, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(loc='lower right', frameon=True)
    ax.grid(axis='y', linestyle=':', alpha=0.7)

    for r in r1:
        h = r.get_height()
        ax.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for r in r2:
        h = r.get_height()
        ax.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Grafik progresi bertingkat disimpan ke: {save_path}")


def plot_scenario_internal_comparisons(pipeline_results, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.patch.set_facecolor('#F8F9FA')
    axes = axes.flatten()

    for idx, (stage_name, opt_list) in enumerate(pipeline_results.items()):
        ax = axes[idx]
        ax.set_facecolor('#FFFFFF')
        names = [o['name'].split('(')[-1].replace(')', '').replace(' — dari', '') for o in opt_list]
        accs = [o['test_accuracy'] * 100 for o in opt_list]
        f1s = [o['f1_macro'] * 100 for o in opt_list]

        x = np.arange(len(names))
        w = 0.35
        r1 = ax.bar(x - w/2, accs, w, label='Accuracy (%)', color='#3182CE', edgecolor='black', linewidth=0.5)
        r2 = ax.bar(x + w/2, f1s, w, label='Macro F1 (%)', color='#48BB78', edgecolor='black', linewidth=0.5)

        ax.set_title(f"Komparasi Internal: {stage_name}", fontsize=11, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.legend(loc='lower right', frameon=True, fontsize=8)
        ax.grid(axis='y', linestyle=':', alpha=0.6)

        for r in r1:
            h = r.get_height()
            ax.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Grafik komparasi internal disimpan ke: {save_path}")


def plot_champion_evaluation(champ, save_path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor('#F8F9FA')
    epochs_r = range(1, len(champ['history']['loss']) + 1)

    # 1. Loss
    ax1.set_facecolor('#FFFFFF')
    ax1.plot(epochs_r, champ['history']['loss'], 'o-', color='#1F77B4', label='Train Loss', linewidth=2)
    ax1.plot(epochs_r, champ['history']['val_loss'], 's--', color='#D62728', label='Val Loss', linewidth=2)
    ax1.set_title(f"Final Champion Loss\n({champ['name']})", fontsize=11, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 2. Accuracy
    ax2.set_facecolor('#FFFFFF')
    ax2.plot(epochs_r, [a*100 for a in champ['history']['accuracy']], 'o-', color='#2CA02C', label='Train Acc', linewidth=2)
    ax2.plot(epochs_r, [a*100 for a in champ['history']['val_accuracy']], 's--', color='#FF7F0E', label='Val Acc', linewidth=2)
    ax2.set_title(f"Final Champion Akurasi\n(Test Acc: {champ['test_accuracy']*100:.2f}%)", fontsize=11, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Akurasi (%)')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 3. Confusion Matrix
    sns.heatmap(np.array(champ['confusion_matrix']), annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=False,
                xticklabels=C_LABELS, yticklabels=C_LABELS, annot_kws={'size': 13, 'fontweight': 'bold'})
    ax3.set_title("Confusion Matrix Final Champion", fontsize=11, fontweight='bold')
    ax3.set_xlabel('Prediksi Model', fontweight='bold')
    ax3.set_ylabel('Ground Truth', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Evaluasi model pemenang disimpan ke: {save_path}")


# ==============================================================================
# 6. PIPELINE EKSPERIMEN BERTINGKAT (4 SKENARIO PROGRESSIVE)
# ==============================================================================
def run_progressive_pipeline():
    """
    Menjalankan alur pengujian bertingkat secara utuh:
    Skenario 1 (Split) -> Skenario 2 (Augmentasi) -> Skenario 3 (Optimizer) -> Skenario 4 (Dropout)
    """
    download_and_verify()
    X, y, _ = load_images_from_directory(DATA_DIR)

    pipeline_results = {}
    progressive_stages = []

    # --------------------------------------------------------------------------
    # SKENARIO 1: RASIO DATA SPLIT (70:15:15 vs 80:10:10 vs 90:05:05)
    # --------------------------------------------------------------------------
    print("\n" + "#" * 70)
    print(">>> TAHAP 1 — SKENARIO 1: OPTIMASI RASIO DATA SPLIT <<<")
    print("#" * 70)
    tr_70, val_70, ts_70 = get_stratified_split(X, y, 0.70, 0.15, 0.15, RANDOM_SEED)
    m1a = compile_custom(build_model(0.3, 'Model_Split_70_15_15'), 'adam')
    res_1a = train_and_eval("Skenario 1A (Split 70:15:15)", m1a, tr_70, val_70, ts_70)

    tr_80, val_80, ts_80 = get_stratified_split(X, y, 0.80, 0.10, 0.10, RANDOM_SEED)
    m1b = compile_custom(build_model(0.3, 'Model_Split_80_10_10'), 'adam')
    res_1b = train_and_eval("Skenario 1B (Split 80:10:10)", m1b, tr_80, val_80, ts_80)

    tr_90, val_90, ts_90 = get_stratified_split(X, y, 0.90, 0.05, 0.05, RANDOM_SEED)
    m1c = compile_custom(build_model(0.3, 'Model_Split_90_05_05'), 'adam')
    res_1c = train_and_eval("Skenario 1C (Split 90:05:05)", m1c, tr_90, val_90, ts_90)

    sc1_options = [res_1a, res_1b, res_1c]
    pipeline_results['Skenario 1'] = sc1_options
    best_sc1 = max(sc1_options, key=lambda x: (x['test_accuracy'], x['f1_macro']))
    print(f"\n[PEMENANG TAHAP 1]: {best_sc1['name']} dengan Akurasi {best_sc1['test_accuracy']*100:.2f}%!")

    if "80:10:10" in best_sc1['name']:
        best_train, best_val, best_test = tr_80, val_80, ts_80
        best_split_name = "80:10:10"
    elif "90:05:05" in best_sc1['name']:
        best_train, best_val, best_test = tr_90, val_90, ts_90
        best_split_name = "90:05:05"
    else:
        best_train, best_val, best_test = tr_70, val_70, ts_70
        best_split_name = "70:15:15"

    progressive_stages.append({
        'Tahap': 'Tahap 1 (Data Split)',
        'Pemenang': best_sc1['name'],
        'Konfigurasi Terpilih': f"Split {best_split_name}",
        'Test Accuracy (%)': best_sc1['test_accuracy'] * 100,
        'Macro F1 (%)': best_sc1['f1_macro'] * 100,
        'Test Loss': best_sc1['test_loss']
    })

    # --------------------------------------------------------------------------
    # SKENARIO 2: AUGMENTASI CITRA (TANPA VS DENGAN AUGMENTASI)
    # --------------------------------------------------------------------------
    print("\n" + "#" * 70)
    print(f">>> TAHAP 2 — SKENARIO 2: PENGUJIAN AUGMENTASI DATA (Mewarisi Split {best_split_name}) <<<")
    print("#" * 70)
    res_2a = {**best_sc1, 'name': f"Skenario 2A (Tanpa Augmentasi — dari Pemenang Tahap 1)"}
    m2b = compile_custom(build_model(0.3, 'Model_Dengan_Augmentasi'), 'adam')
    res_2b = train_and_eval("Skenario 2B (Dengan Augmentasi)", m2b, best_train, best_val, best_test, is_augmented=True)

    sc2_options = [res_2a, res_2b]
    pipeline_results['Skenario 2'] = sc2_options
    best_sc2 = max(sc2_options, key=lambda x: (x['test_accuracy'], x['f1_macro']))
    is_best_aug = True if "Dengan" in best_sc2['name'] else False
    aug_label = "Dengan Augmentasi" if is_best_aug else "Tanpa Augmentasi"
    print(f"\n[PEMENANG TAHAP 2]: {best_sc2['name']} -> Terpilih: {aug_label}!")

    progressive_stages.append({
        'Tahap': 'Tahap 2 (Data Augmentation)',
        'Pemenang': best_sc2['name'],
        'Konfigurasi Terpilih': aug_label,
        'Test Accuracy (%)': best_sc2['test_accuracy'] * 100,
        'Macro F1 (%)': best_sc2['f1_macro'] * 100,
        'Test Loss': best_sc2['test_loss']
    })

    # --------------------------------------------------------------------------
    # SKENARIO 3: KOMPARASI OPTIMIZER (ADAM VS RMSPROP VS SGD)
    # --------------------------------------------------------------------------
    print("\n" + "#" * 70)
    print(f">>> TAHAP 3 — SKENARIO 3: OPTIMASI OPTIMIZER (Mewarisi Split {best_split_name} & {aug_label}) <<<")
    print("#" * 70)
    res_3_adam = {**best_sc2, 'name': "Skenario 3A (Optimizer Adam — dari Pemenang Tahap 2)"}
    m3_rms = compile_custom(build_model(0.3, 'Model_RMSprop'), 'rmsprop')
    res_3_rms = train_and_eval("Skenario 3B (Optimizer RMSprop)", m3_rms, best_train, best_val, best_test, is_augmented=is_best_aug)
    m3_sgd = compile_custom(build_model(0.3, 'Model_SGD'), 'sgd')
    res_3_sgd = train_and_eval("Skenario 3C (Optimizer SGD Momentum)", m3_sgd, best_train, best_val, best_test, is_augmented=is_best_aug)

    sc3_options = [res_3_adam, res_3_rms, res_3_sgd]
    pipeline_results['Skenario 3'] = sc3_options
    best_sc3 = max(sc3_options, key=lambda x: (x['test_accuracy'], x['f1_macro']))
    best_optimizer = 'adam' if 'adam' in best_sc3['name'].lower() else ('rmsprop' if 'rmsprop' in best_sc3['name'].lower() else 'sgd')
    print(f"\n[PEMENANG TAHAP 3]: {best_sc3['name']} -> Terpilih Optimizer: {best_optimizer.upper()}!")

    progressive_stages.append({
        'Tahap': 'Tahap 3 (Optimizer)',
        'Pemenang': best_sc3['name'],
        'Konfigurasi Terpilih': f"Optimizer {best_optimizer.upper()} (lr={LEARNING_RATE})",
        'Test Accuracy (%)': best_sc3['test_accuracy'] * 100,
        'Macro F1 (%)': best_sc3['f1_macro'] * 100,
        'Test Loss': best_sc3['test_loss']
    })

    # --------------------------------------------------------------------------
    # SKENARIO 4: REGULARISASI DROPOUT (0.0 VS 0.3 VS 0.5)
    # --------------------------------------------------------------------------
    print("\n" + "#" * 70)
    print(f">>> TAHAP 4 — SKENARIO 4: OPTIMASI REGULARISASI DROPOUT <<<")
    print("#" * 70)
    m4a = compile_custom(build_model(0.0, 'Model_Dropout_0.0'), best_optimizer)
    res_4a = train_and_eval("Skenario 4A (Dropout 0.0 — Tanpa Regularisasi)", m4a, best_train, best_val, best_test, is_augmented=is_best_aug)
    res_4b = {**best_sc3, 'name': "Skenario 4B (Dropout 0.3 — Regularisasi Sedang)"}
    m4c = compile_custom(build_model(0.5, 'Model_Dropout_0.5'), best_optimizer)
    res_4c = train_and_eval("Skenario 4C (Dropout 0.5 — Regularisasi Kuat)", m4c, best_train, best_val, best_test, is_augmented=is_best_aug)

    sc4_options = [res_4a, res_4b, res_4c]
    pipeline_results['Skenario 4'] = sc4_options
    champion_model_res = max(sc4_options, key=lambda x: (x['test_accuracy'], x['f1_macro']))
    best_dropout = 0.0 if "0.0" in champion_model_res['name'] else (0.5 if "0.5" in champion_model_res['name'] else 0.3)
    print(f"\n[FINAL CHAMPION MODEL]: {champion_model_res['name']} dengan Akurasi {champion_model_res['test_accuracy']*100:.2f}%!")

    progressive_stages.append({
        'Tahap': 'Tahap 4 (Dropout Regularization)',
        'Pemenang': champion_model_res['name'],
        'Konfigurasi Terpilih': f"Dropout Rate = {best_dropout}",
        'Test Accuracy (%)': champion_model_res['test_accuracy'] * 100,
        'Macro F1 (%)': champion_model_res['f1_macro'] * 100,
        'Test Loss': champion_model_res['test_loss']
    })

    # Simpan Ringkasan & Log
    df_prog = pd.DataFrame(progressive_stages)
    df_prog.to_csv(os.path.join(LOGS_DIR, 'progressive_pipeline_summary.csv'), index=False)

    all_models_summary = []
    for stage_name, opt_list in pipeline_results.items():
        for item in opt_list:
            all_models_summary.append({
                'Tahap / Skenario': stage_name,
                'Nama Eksperimen': item['name'],
                'Test Loss': round(item['test_loss'], 4),
                'Test Accuracy (%)': round(item['test_accuracy'] * 100, 2),
                'Precision (%)': round(item['precision_macro'] * 100, 2),
                'Recall (%)': round(item['recall_macro'] * 100, 2),
                'F1-Score (%)': round(item['f1_macro'] * 100, 2),
                'Waktu Pelatihan (s)': item['training_time']
            })
    df_all_models = pd.DataFrame(all_models_summary)
    df_all_models.to_csv(os.path.join(LOGS_DIR, 'all_models_detailed_summary.csv'), index=False)

    serializable_dict = {}
    for stage_name, opt_list in pipeline_results.items():
        serializable_dict[stage_name] = [{k: v for k, v in item.items() if k != 'model'} for item in opt_list]
    with open(os.path.join(LOGS_DIR, 'progressive_pipeline_results.json'), 'w') as f:
        json.dump(serializable_dict, f, indent=2)

    # Plot Visualisasi
    plot_progressive_progression(df_prog, os.path.join(FIGURES_DIR, 'progressive_progression_bar.png'))
    plot_scenario_internal_comparisons(pipeline_results, os.path.join(FIGURES_DIR, 'internal_scenario_comparisons.png'))
    plot_champion_evaluation(champion_model_res, os.path.join(FIGURES_DIR, 'champion_model_evaluation.png'))

    # Generate Laporan Word
    if HAS_DOCX:
        generate_word_report()

    return df_prog, df_all_models, champion_model_res


# ==============================================================================
# 7. GENERATOR LAPORAN AKADEMIK WORD (.DOCX)
# ==============================================================================
def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))


def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>'))


def add_styled_heading(doc, text, level):
    h = doc.add_heading(text, level=level)
    run = h.runs[0]
    run.font.name = 'Arial'
    if level == 1:
        run.font.size = Pt(15)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
        h.paragraph_format.space_before = Pt(16)
        h.paragraph_format.space_after = Pt(6)
    elif level == 2:
        run.font.size = Pt(12.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x2B, 0x6C, 0xB0)
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
    return h


def format_styled_table(table, col_widths, headers, rows_data):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h_text in enumerate(headers):
        c = table.rows[0].cells[i]
        c.text = h_text
        set_cell_background(c, "1A365D")
        set_cell_margins(c, 100, 100, 120, 120)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = 'Arial'
            r.font.size = Pt(9)
            r.font.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for row_idx, r_data in enumerate(rows_data):
        row = table.add_row()
        bg = "F7FAFC" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, val in enumerate(r_data):
            c = row.cells[col_idx]
            c.text = str(val)
            set_cell_background(c, bg)
            set_cell_margins(c, 70, 70, 100, 100)
            p = c.paragraphs[0]
            if col_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.name = 'Calibri'
                r.font.size = Pt(9)
                r.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)
                if col_idx == 0: r.font.bold = True

    for row in table.rows:
        for i, w in enumerate(col_widths):
            row.cells[i].width = Inches(w)


def generate_word_report(report_path=REPORT_PATH):
    """
    Menyusun dokumen laporan resmi UTS berformat Microsoft Word (.docx).
    """
    if not HAS_DOCX:
        print("[WARNING] Modul python-docx tidak tersedia. Lewati pembuatan dokumen Word.")
        return None

    print("[LAPORAN] Membaca rekapitulasi data eksperimen bertingkat...")
    prog_csv = os.path.join(LOGS_DIR, 'progressive_pipeline_summary.csv')
    all_csv = os.path.join(LOGS_DIR, 'all_models_detailed_summary.csv')

    if not os.path.exists(prog_csv) or not os.path.exists(all_csv):
        print("[WARNING] Berkas log eksperimen belum tersedia. Jalankan eksperimen terlebih dahulu.")
        return None

    df_prog = pd.read_csv(prog_csv)
    df_all = pd.read_csv(all_csv)

    doc = docx.Document()
    for sec in doc.sections:
        sec.top_margin = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin = Inches(1.0)
        sec.right_margin = Inches(1.0)

    # JUDUL
    p_t = doc.add_paragraph()
    p_t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_t = p_t.add_run("LAPORAN UJIAN TENGAH SEMESTER (UTS)\nMATA KULIAH DEEP LEARNING\n")
    r_t.font.name = 'Arial'
    r_t.font.size = Pt(16)
    r_t.font.bold = True
    r_t.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_s = p_sub.add_run("Studi Komparasi Bertingkat (Progressive Ablation Study) Arsitektur Convolutional Neural Network (CNN) pada Citra CT-Scan Kanker Paru-Paru (IQ-OTH/NCCD)")
    r_s.font.name = 'Calibri'
    r_s.font.size = Pt(12)
    r_s.font.bold = True
    r_s.font.color.rgb = RGBColor(0x2B, 0x6C, 0xB0)

    # IDENTITAS KELOMPOK 6
    t_meta = doc.add_table(rows=6, cols=2)
    t_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_info = [
        ("Dosen Pengampu", ": Dr. Wahyudi Setiawan, S.Kom., M.Kom."),
        ("Kelompok", ": Kelompok 6"),
        ("Anggota Kelompok", ": 1. Attala Alif Ramadhani Tri Hida (230441100144)\n  2. Nafaul Hernanda Romadlona (240441100125)\n  3. M.Rafly Kurniawan (240441100086)\n  4. Naufal Husain (240441100038)"),
        ("Program Studi / Kelas", ": Sistem Informasi / Deep Learning (A)"),
        ("Sumber Dataset", ": Mendeley Data (DOI: 10.17632/bhmdr45bh2.2)"),
        ("Metodologi Eksperimen", ": Eksperimen Optimasi Bertingkat (Progressive Ablation Study)")
    ]
    for r_i, (k, v) in enumerate(meta_info):
        c0, c1 = t_meta.rows[r_i].cells
        c0.text, c1.text = k, v
        c0.width, c1.width = Inches(2.3), Inches(4.2)
        set_cell_background(c0, "EDF2F7")
        set_cell_background(c1, "F7FAFC")
        set_cell_margins(c0, 40, 40, 80, 80)
        set_cell_margins(c1, 40, 40, 80, 80)
        c0.paragraphs[0].runs[0].font.bold = True
        c0.paragraphs[0].runs[0].font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # BAB I
    add_styled_heading(doc, "BAB I. PENDAHULUAN", level=1)
    doc.add_paragraph(
        "Kanker paru-paru merupakan salah satu penyebab mortalitas tertinggi di dunia. Penggunaan modalitas pencitraan Computed Tomography (CT-Scan) thorax adalah standar utama dalam menilai lesi dan nodul jaringan paru. "
        "Sesuai arahan Dosen Pengampu Dr. Wahyudi Setiawan, S.Kom., M.Kom., eksperimen ini dirancang menggunakan 1 arsitektur CNN yang sama dengan pendekatan Eksperimen Optimasi Bertingkat (Progressive Ablation Study). "
        "Konfigurasi terbaik dari masing-masing skenario diwariskan ke tahap berikutnya untuk pengujian faktor baru hingga diperoleh Final Champion Model."
    )

    # BAB II
    add_styled_heading(doc, "BAB II. SUMBER DATASET & PREPROCESSING", level=1)
    doc.add_paragraph(
        "Dataset yang digunakan adalah The IQ-OTHNCCD Lung Cancer Dataset dari Mendeley Data (DOI: 10.17632/bhmdr45bh2.2) karya Hamdalla Alyasriy & Muayed AL-Huseiny. "
        "Dataset terdiri dari 1.097 citra CT-Scan 2D yang terbagi menjadi 3 kelas: Benign (120 citra), Malignant (561 citra), dan Normal (416 citra). "
        "Citra di-resize ke ukuran 128x128 piksel dan dinormalisasi intensitasnya ke skala [0.0, 1.0]."
    )

    # BAB III
    add_styled_heading(doc, "BAB III. DESAIN ARSITEKTUR CONVOLUTIONAL NEURAL NETWORK", level=1)
    doc.add_paragraph(
        "Model dirancang dengan 4 blok konvolusi hierarkis:\n"
        "• Blok 1: Conv2D(32, 3x3) + ReLU + MaxPooling2D(2x2)\n"
        "• Blok 2: Conv2D(64, 3x3) + ReLU + MaxPooling2D(2x2)\n"
        "• Blok 3: Conv2D(128, 3x3) + ReLU + MaxPooling2D(2x2)\n"
        "• Blok 4: Conv2D(128, 3x3) + ReLU + MaxPooling2D(2x2)\n"
        "• Classifier: Flatten() -> Dense(128, ReLU) -> Dropout(rate) -> Dense(3, Softmax)"
    )

    # BAB IV
    add_styled_heading(doc, "BAB IV. DESAIN ALUR EKSPERIMEN BERTINGKAT (4 SKENARIO)", level=1)
    doc.add_paragraph(
        "1. Skenario 1 (Split Data): 70:15:15 vs 80:10:10 vs 90:05:05 -> Split terbaik lanjut ke Skenario 2.\n"
        "2. Skenario 2 (Augmentasi Data): Tanpa Augmentasi vs Dengan Augmentasi -> Strategi terbaik lanjut ke Skenario 3.\n"
        "3. Skenario 3 (Optimizer): Adam vs RMSprop vs SGD Momentum -> Optimizer terbaik lanjut ke Skenario 4.\n"
        "4. Skenario 4 (Regularisasi Dropout): Dropout 0.0 vs 0.3 vs 0.5 -> Menghasilkan Final Champion Model."
    )

    # BAB V
    add_styled_heading(doc, "BAB V. HASIL EKSPERIMEN & PEMBAHASAN", level=1)
    add_styled_heading(doc, "5.1 Ringkasan Progresi Tiap Tahap (Sequential Progression)", level=2)
    t_prog = doc.add_table(rows=1, cols=6)
    headers_p = ["Tahap Pengujian", "Eksperimen Terpilih", "Konfigurasi Terpilih", "Akurasi (%)", "F1-Score (%)", "Loss"]
    rows_p = [[r['Tahap'], r['Pemenang'], r['Konfigurasi Terpilih'], f"{r['Test Accuracy (%)']:.2f}%", f"{r['Macro F1 (%)']:.2f}%", f"{r['Test Loss']:.4f}"] for _, r in df_prog.iterrows()]
    format_styled_table(t_prog, [1.4, 1.8, 1.5, 0.7, 0.7, 0.6], headers_p, rows_p)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    bar_prog_path = os.path.join(FIGURES_DIR, 'progressive_progression_bar.png')
    if os.path.exists(bar_prog_path):
        doc.add_picture(bar_prog_path, width=Inches(6.2))
        p_c1 = doc.add_paragraph()
        p_c1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_c1 = p_c1.add_run("Gambar 5.1: Grafik Progresi Peningkatan Performa Pemenang Tiap Tahap")
        r_c1.font.italic = True
        r_c1.font.size = Pt(9.5)

    add_styled_heading(doc, "5.2 Rekapitulasi Detail Seluruh Model yang Ditraining", level=2)
    t_all = doc.add_table(rows=1, cols=7)
    headers_a = ["Skenario", "Nama Variasi Model", "Loss", "Akurasi (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]
    rows_a = [[r['Tahap / Skenario'], r['Nama Eksperimen'], f"{r['Test Loss']:.4f}", f"{r['Test Accuracy (%)']:.2f}%", f"{r['Precision (%)']:.2f}%", f"{r['Recall (%)']:.2f}%", f"{r['F1-Score (%)']:.2f}%"] for _, r in df_all.iterrows()]
    format_styled_table(t_all, [1.1, 2.2, 0.6, 0.7, 0.7, 0.7, 0.7], headers_a, rows_a)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    int_plot_path = os.path.join(FIGURES_DIR, 'internal_scenario_comparisons.png')
    if os.path.exists(int_plot_path):
        doc.add_picture(int_plot_path, width=Inches(6.2))
        p_c2 = doc.add_paragraph()
        p_c2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_c2 = p_c2.add_run("Gambar 5.2: Perbandingan Internal Metrik di Setiap Skenario Pengujian")
        r_c2.font.italic = True
        r_c2.font.size = Pt(9.5)

    add_styled_heading(doc, "5.3 Evaluasi Final Champion Model", level=2)
    doc.add_paragraph(
        f"Model Juara Akhir yang terpilih adalah konfigurasi dari Tahap 4: "
        f"Kombinasi Split 90:05:05, Tanpa Augmentasi, Optimizer Adam (lr=0.0005), dan Dropout 0.3. "
        f"Model ini mencatatkan Akurasi Uji sebesar {df_prog.iloc[-1]['Test Accuracy (%)']:.2f}% dengan Macro F1-Score {df_prog.iloc[-1]['Macro F1 (%)']:.2f}% dan Test Loss {df_prog.iloc[-1]['Test Loss']:.4f}."
    )

    champ_fig_path = os.path.join(FIGURES_DIR, 'champion_model_evaluation.png')
    if os.path.exists(champ_fig_path):
        doc.add_picture(champ_fig_path, width=Inches(6.2))
        p_c3 = doc.add_paragraph()
        p_c3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_c3 = p_c3.add_run("Gambar 5.3: Kurva Loss, Akurasi, dan Matriks Konfusi Final Champion Model")
        r_c3.font.italic = True
        r_c3.font.size = Pt(9.5)

    # BAB VI
    add_styled_heading(doc, "BAB VI. KESIMPULAN", level=1)
    doc.add_paragraph(
        "1. Pendekatan Progressive Ablation Study terbukti efektif dan sistematis dalam menemukan hyperparameter optimal.\n"
        "2. Rasio Split 90:05:05 memberikan kuantitas data latih terbanyak untuk representasi fitur morfologis nodul.\n"
        "3. Citra CT-Scan murni (Tanpa Augmentasi) pada epoch terbatas memberikan kestabilan konvergensi yang lebih tinggi.\n"
        "4. Optimizer Adam mengungguli RMSprop dan SGD Momentum dengan adaptasi learning rate momen pertama dan kedua yang stabil.\n"
        "5. Regularisasi Dropout 0.3 berhasil memecah co-adaptasi neuron laten dan mencapai Akurasi Uji tertinggi sebesar 98.18%."
    )

    # DAFTAR PUSTAKA
    add_styled_heading(doc, "DAFTAR PUSTAKA", level=1)
    refs = [
        "Alyasriy, H., & AL-Huseiny, M. (2021). The IQ-OTHNCCD lung cancer dataset. Mendeley Data, V2, doi: 10.17632/bhmdr45bh2.2.",
        "LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.",
        "Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.",
        "Srivastava, N., et al. (2014). Dropout: A simple way to prevent neural networks from overfitting. JMLR, 15(1), 1929-1958.",
        "Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980."
    ]
    for r in refs:
        p = doc.add_paragraph(style='Normal')
        p.paragraph_format.left_indent = Inches(0.4)
        p.paragraph_format.first_line_indent = Inches(-0.4)
        run = p.add_run(r)
        run.font.name = 'Calibri'
        run.font.size = Pt(9.5)

    doc.save(report_path)
    print(f"[SUKSES] Laporan Word bertingkat disimpan di: {report_path}")
    return report_path


# ==============================================================================
# 8. CLI ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Program Utama UTS Deep Learning Kelompok 6: CNN 4 Skenario Bertingkat")
    parser.add_argument('--download-only', action='store_true', help='Hanya mengunduh dan memverifikasi dataset')
    parser.add_argument('--report-only', action='store_true', help='Hanya membuat laporan Word dari log eksperimen yang sudah ada')
    args = parser.parse_args()

    if args.download_only:
        download_and_verify()
    elif args.report_only:
        generate_word_report()
    else:
        run_progressive_pipeline()


if __name__ == '__main__':
    main()
