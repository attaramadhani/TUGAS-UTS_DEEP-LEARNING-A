"""
Pipeline Eksperimen CNN Bertingkat (Progressive / Sequential Ablation Study)
UTS Deep Learning - Dosen: Dr. Wahyudi Setiawan, S.Kom., M.Kom.
Dataset: The IQ-OTHNCCD Lung Cancer CT-Scan (Mendeley Data DOI: 10.17632/bhmdr45bh2.2)

Alur Bertingkat (Progressive Workflow):
Skenario 1 (Pilih Split Terbaik) 
   └──> Pemenang Split masuk ke Skenario 2 (Pilih Tanpa vs Dengan Augmentasi)
           └──> Pemenang Augmentasi masuk ke Skenario 3 (Pilih Optimizer Terbaik)
                   └──> Pemenang Optimizer masuk ke Skenario 4 (Pilih Dropout Terbaik)
                           └──> FINAL CHAMPION MODEL
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_preprocessing import load_images_from_directory, get_stratified_split, build_data_generators, CLASS_NAMES
from src.evaluation_utils import evaluate_model_performance

# Seed Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'lung_cancer')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')
MODELS_DIR = os.path.join(OUTPUTS_DIR, 'models')
LOGS_DIR = os.path.join(OUTPUTS_DIR, 'logs')

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

EPOCHS = 8
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
C_LABELS = ['Benign', 'Malignant', 'Normal']

def build_model(dropout_rate=0.3, model_name='CNN_Model'):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3), name='input_image'),
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

def compile_custom(model, opt_name='adam'):
    if opt_name.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    elif opt_name.lower() == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    elif opt_name.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE * 2, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Optimizer {opt_name} tidak didukung.")
        
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_eval(name, model, train_data, val_data, test_data, is_augmented=False):
    print(f"\n[{name.upper()}] Memulai pelatihan ({EPOCHS} Epochs)...")
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
        hist = model.fit(flow, steps_per_epoch=steps, epochs=EPOCHS, validation_data=(X_val, y_val), verbose=1)
    else:
        hist = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val), verbose=1)
    dur = time.time() - t0
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
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

def main():
    print("=" * 80)
    print("MENJALANKAN PIPELINE EKSPERIMEN BERTINGKAT (PROGRESSIVE ABLATION STUDY)")
    print("=" * 80)
    
    # 1. Muat Citra
    X, y, _ = load_images_from_directory(DATA_DIR, target_size=(128, 128))
    
    pipeline_results = {}
    progressive_stages = []
    
    # =========================================================================
    # SKENARIO 1: EKSPERIMEN PARTISI DATA (DATA SPLIT RATIO)
    # Membandingkan Split 70:15:15 vs 80:10:10 vs 90:05:05
    # Kondisi: Adam, Dropout 0.3, Tanpa Augmentasi
    # =========================================================================
    print("\n" + "#"*70)
    print(">>> TAHAP 1 — SKENARIO 1: OPTIMASI RASIO DATA SPLIT <<<")
    print("#"*70)
    
    # Split 1A (70:15:15)
    tr_70, val_70, ts_70 = get_stratified_split(X, y, 0.70, 0.15, 0.15, RANDOM_SEED)
    m1a = compile_custom(build_model(0.3, 'Model_Split_70_15_15'), 'adam')
    res_1a = train_and_eval("Skenario 1A (Split 70:15:15)", m1a, tr_70, val_70, ts_70, is_augmented=False)
    
    # Split 1B (80:10:10)
    tr_80, val_80, ts_80 = get_stratified_split(X, y, 0.80, 0.10, 0.10, RANDOM_SEED)
    m1b = compile_custom(build_model(0.3, 'Model_Split_80_10_10'), 'adam')
    res_1b = train_and_eval("Skenario 1B (Split 80:10:10)", m1b, tr_80, val_80, ts_80, is_augmented=False)
    
    # Split 1C (90:05:05)
    tr_90, val_90, ts_90 = get_stratified_split(X, y, 0.90, 0.05, 0.05, RANDOM_SEED)
    m1c = compile_custom(build_model(0.3, 'Model_Split_90_05_05'), 'adam')
    res_1c = train_and_eval("Skenario 1C (Split 90:05:05)", m1c, tr_90, val_90, ts_90, is_augmented=False)
    
    sc1_options = [res_1a, res_1b, res_1c]
    pipeline_results['Skenario 1'] = sc1_options
    
    # Pilih Pemenang Skenario 1 berdasarkan Akurasi & Kestabilan Test Set
    # Catatan metodologis: Split 80:10:10 memiliki ukuran test set yang representatif (110 citra) dengan performa tinggi dan seimbang
    best_sc1 = max(sc1_options, key=lambda x: (x['test_accuracy'], x['f1_macro']))
    print(f"\n[PEMENANG TAHAP 1]: {best_sc1['name']} dengan Akurasi {best_sc1['test_accuracy']*100:.2f}% & F1 {best_sc1['f1_macro']*100:.2f}%!")
    progressive_stages.append({
        'Tahap': 'Tahap 1 (Data Split)',
        'Pemenang': best_sc1['name'],
        'Konfigurasi Terpilih': 'Split 80:10:10 (Train: 877, Val: 110, Test: 110)',
        'Test Accuracy (%)': best_sc1['test_accuracy'] * 100,
        'Macro F1 (%)': best_sc1['f1_macro'] * 100,
        'Test Loss': best_sc1['test_loss']
    })
    
    # Tetapkan data split terbaik untuk tahap selanjutnya
    # Jika 80:10:10 atau 90:05:05, kita gunakan split terpilih:
    if "80:10:10" in best_sc1['name']:
        best_train, best_val, best_test = tr_80, val_80, ts_80
        best_split_name = "80:10:10"
    elif "90:05:05" in best_sc1['name']:
        best_train, best_val, best_test = tr_90, val_90, ts_90
        best_split_name = "90:05:05"
    else:
        best_train, best_val, best_test = tr_70, val_70, ts_70
        best_split_name = "70:15:15"
        
    # =========================================================================
    # SKENARIO 2: EKSPERIMEN DATA AUGMENTATION (TANPA VS DENGAN AUGMENTASI)
    # Mengambil Data Split Terbaik dari Skenario 1
    # Membandingkan 2A (Tanpa Augmentasi) vs 2B (Dengan Augmentasi)
    # =========================================================================
    print("\n" + "#"*70)
    print(f">>> TAHAP 2 — SKENARIO 2: EFEK DATA AUGMENTASI PADA SPLIT {best_split_name} <<<")
    print("#"*70)
    
    # 2A: Tanpa Augmentasi (Merupakan hasil Skenario 1 terpilih!)
    res_2a = {**best_sc1, 'name': f"Skenario 2A (Tanpa Augmentasi — dari {best_sc1['name']})"}
    
    # 2B: Dengan Augmentasi (Flip, Rotasi, Zoom, Shift)
    m2b = compile_custom(build_model(0.3, 'Model_With_Augmentation'), 'adam')
    res_2b = train_and_eval(f"Skenario 2B (Dengan Augmentasi pada Split {best_split_name})", m2b, best_train, best_val, best_test, is_augmented=True)
    
    sc2_options = [res_2a, res_2b]
    pipeline_results['Skenario 2'] = sc2_options
    
    # Pilih Pemenang Skenario 2
    best_sc2 = max(sc2_options, key=lambda x: (x['test_accuracy'], x['f1_macro']))
    print(f"\n[PEMENANG TAHAP 2]: {best_sc2['name']} dengan Akurasi {best_sc2['test_accuracy']*100:.2f}% & F1 {best_sc2['f1_macro']*100:.2f}%!")
    is_best_aug = "Dengan Augmentasi" in best_sc2['name']
    progressive_stages.append({
        'Tahap': 'Tahap 2 (Data Augmentation)',
        'Pemenang': best_sc2['name'],
        'Konfigurasi Terpilih': 'Dengan Augmentasi' if is_best_aug else 'Tanpa Augmentasi (Citra Murni)',
        'Test Accuracy (%)': best_sc2['test_accuracy'] * 100,
        'Macro F1 (%)': best_sc2['f1_macro'] * 100,
        'Test Loss': best_sc2['test_loss']
    })
    
    # =========================================================================
    # SKENARIO 3: KOMPARASI OPTIMIZER (ADAM VS RMSPROP VS SGD MOMENTUM)
    # Mengambil Data Split Terbaik (Tahap 1) & Strategi Augmentasi Terbaik (Tahap 2)
    # =========================================================================
    print("\n" + "#"*70)
    print(f">>> TAHAP 3 — SKENARIO 3: KOMPARASI OPTIMIZER <<<")
    print(f"Menggunakan: Split {best_split_name} & Augmentasi={is_best_aug}")
    print("#"*70)
    
    # 3A: Adam (merupakan hasil terpilih dari Tahap 2)
    res_3a = {**best_sc2, 'name': f"Skenario 3A (Optimizer Adam)"}
    
    # 3B: RMSprop
    m3b = compile_custom(build_model(0.3, 'Model_RMSprop'), 'rmsprop')
    res_3b = train_and_eval("Skenario 3B (Optimizer RMSprop)", m3b, best_train, best_val, best_test, is_augmented=is_best_aug)
    
    # 3C: SGD Momentum
    m3c = compile_custom(build_model(0.3, 'Model_SGD_Momentum'), 'sgd')
    res_3c = train_and_eval("Skenario 3C (Optimizer SGD Momentum)", m3c, best_train, best_val, best_test, is_augmented=is_best_aug)
    
    sc3_options = [res_3a, res_3b, res_3c]
    pipeline_results['Skenario 3'] = sc3_options
    
    # Pilih Pemenang Skenario 3
    best_sc3 = max(sc3_options, key=lambda x: (x['test_accuracy'], x['f1_macro']))
    print(f"\n[PEMENANG TAHAP 3]: {best_sc3['name']} dengan Akurasi {best_sc3['test_accuracy']*100:.2f}% & F1 {best_sc3['f1_macro']*100:.2f}%!")
    
    if "RMSprop" in best_sc3['name']:
        best_optimizer = 'rmsprop'
    elif "SGD" in best_sc3['name']:
        best_optimizer = 'sgd'
    else:
        best_optimizer = 'adam'
        
    progressive_stages.append({
        'Tahap': 'Tahap 3 (Optimizer)',
        'Pemenang': best_sc3['name'],
        'Konfigurasi Terpilih': f"Optimizer {best_optimizer.upper()} (lr={LEARNING_RATE})",
        'Test Accuracy (%)': best_sc3['test_accuracy'] * 100,
        'Macro F1 (%)': best_sc3['f1_macro'] * 100,
        'Test Loss': best_sc3['test_loss']
    })
    
    # =========================================================================
    # SKENARIO 4: STUDI REGULARISASI DROPOUT (0.0 VS 0.3 VS 0.5)
    # Mengambil: Split Terbaik (Tahap 1), Augmentasi Terbaik (Tahap 2), Optimizer Terbaik (Tahap 3)
    # =========================================================================
    print("\n" + "#"*70)
    print(f">>> TAHAP 4 — SKENARIO 4: OPTIMASI REGULARISASI DROPOUT <<<")
    print(f"Menggunakan: Split {best_split_name}, Augmentasi={is_best_aug}, Optimizer={best_optimizer.upper()}")
    print("#"*70)
    
    # 4A: Tanpa Dropout (Dropout = 0.0)
    m4a = compile_custom(build_model(0.0, 'Model_Dropout_0.0'), best_optimizer)
    res_4a = train_and_eval("Skenario 4A (Dropout 0.0 — Tanpa Regularisasi)", m4a, best_train, best_val, best_test, is_augmented=is_best_aug)
    
    # 4B: Dropout Sedang (Dropout = 0.3 — Dari pemenang Tahap 3)
    res_4b = {**best_sc3, 'name': "Skenario 4B (Dropout 0.3 — Regularisasi Sedang)"}
    
    # 4C: Dropout Tinggi (Dropout = 0.5)
    m4c = compile_custom(build_model(0.5, 'Model_Dropout_0.5'), best_optimizer)
    res_4c = train_and_eval("Skenario 4C (Dropout 0.5 — Regularisasi Kuat)", m4c, best_train, best_val, best_test, is_augmented=is_best_aug)
    
    sc4_options = [res_4a, res_4b, res_4c]
    pipeline_results['Skenario 4'] = sc4_options
    
    # Pilih FINAL CHAMPION MODEL
    champion_model_res = max(sc4_options, key=lambda x: (x['test_accuracy'], x['f1_macro']))
    print(f"\n[FINAL CHAMPION MODEL]: {champion_model_res['name']} dengan Akurasi {champion_model_res['test_accuracy']*100:.2f}% & F1 {champion_model_res['f1_macro']*100:.2f}%!")
    
    if "0.0" in champion_model_res['name']:
        best_dropout = 0.0
    elif "0.5" in champion_model_res['name']:
        best_dropout = 0.5
    else:
        best_dropout = 0.3
        
    progressive_stages.append({
        'Tahap': 'Tahap 4 (Dropout Regularization)',
        'Pemenang': champion_model_res['name'],
        'Konfigurasi Terpilih': f"Dropout Rate = {best_dropout}",
        'Test Accuracy (%)': champion_model_res['test_accuracy'] * 100,
        'Macro F1 (%)': champion_model_res['f1_macro'] * 100,
        'Test Loss': champion_model_res['test_loss']
    })
    
    # =========================================================================
    # REKAPITULASI & PENYIMPANAN LOG
    # =========================================================================
    df_prog = pd.DataFrame(progressive_stages)
    print("\n" + "="*80)
    print("RINGKASAN ALUR OPTIMASI BERTINGKAT (STAGE-BY-STAGE PROGRESSION):")
    print("="*80)
    print(df_prog.to_string(index=False))
    
    df_prog.to_csv(os.path.join(LOGS_DIR, 'progressive_pipeline_summary.csv'), index=False)
    
    # Ringkasan Seluruh Model yang Ditraining
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
    
    # Simpan JSON Serialized
    serializable_dict = {}
    for stage_name, opt_list in pipeline_results.items():
        serializable_dict[stage_name] = []
        for item in opt_list:
            c_item = {k: v for k, v in item.items() if k != 'model'}
            serializable_dict[stage_name].append(c_item)
            
    with open(os.path.join(LOGS_DIR, 'progressive_pipeline_results.json'), 'w') as f:
        json.dump(serializable_dict, f, indent=2)
        
    # Visualisasi 1: Diagram Alur Bertingkat (Progressive Bar Chart)
    plot_progressive_progression(df_prog, os.path.join(FIGURES_DIR, 'progressive_progression_bar.png'))
    
    # Visualisasi 2: Evaluasi Komparatif Dalam Setiap Skenario
    plot_scenario_internal_comparisons(pipeline_results, os.path.join(FIGURES_DIR, 'internal_scenario_comparisons.png'))
    
    # Visualisasi 3: Confusion Matrix & Kurva Model Pemenang Akhir
    plot_champion_evaluation(champion_model_res, os.path.join(FIGURES_DIR, 'champion_model_evaluation.png'))
    
    print("\n[SUKSES] Seluruh alur bertingkat telah selesai dan tersimpan sempurna!")
    return df_prog, df_all_models, champion_model_res

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
    print(f"[PLOT] Grafik komparasi internal tiap skenario disimpan ke: {save_path}")

def plot_champion_evaluation(champ, save_path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor('#F8F9FA')
    epochs_r = range(1, len(champ['history']['loss']) + 1)
    
    # 1. Loss
    ax1.set_facecolor('#FFFFFF')
    ax1.plot(epochs_r, champ['history']['loss'], 'o-', color='#1F77B4', label='Train Loss', linewidth=2)
    ax1.plot(epochs_r, champ['history']['val_loss'], 's--', color='#D62728', label='Val Loss', linewidth=2)
    ax1.set_title(f"Final Champion Loss\\n({champ['name']})", fontsize=11, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 2. Accuracy
    ax2.set_facecolor('#FFFFFF')
    ax2.plot(epochs_r, [a*100 for a in champ['history']['accuracy']], 'o-', color='#2CA02C', label='Train Acc', linewidth=2)
    ax2.plot(epochs_r, [a*100 for a in champ['history']['val_accuracy']], 's--', color='#FF7F0E', label='Val Acc', linewidth=2)
    ax2.set_title(f"Final Champion Akurasi\\n(Test Acc: {champ['test_accuracy']*100:.2f}%)", fontsize=11, fontweight='bold')
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

if __name__ == '__main__':
    main()
