"""
Modul Utilitas Evaluasi dan Visualisasi Hasil Eksperimen CNN
Menyediakan kalkulasi metrik (Akurasi, Loss, Precision, Recall, F1-Score),
visualisasi kurva loss/akurasi, confusion matrix, dan perbandingan lintas skenario.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

CLASS_NAMES = ['Bengin', 'Malignant', 'Normal']

def evaluate_model_performance(model, X_test, y_test, class_names=CLASS_NAMES):
    """
    Mengevaluasi model pada test set dan menghitung metrik lengkap.
    """
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    cr_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'precision_macro': float(p_macro),
        'recall_macro': float(r_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(p_weighted),
        'recall_weighted': float(r_weighted),
        'f1_weighted': float(f1_weighted),
        'confusion_matrix': cm.tolist(),
        'classification_report': cr_dict,
        'y_true': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'y_pred_probs': y_pred_probs.tolist()
    }
    return results

def plot_training_history(history, title, save_path):
    """
    Membuat grafik Loss dan Akurasi per Epoch (Train vs Validation).
    """
    epochs = range(1, len(history.history['loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#F8F9FA')
    
    # 1. Plot Loss
    ax1.set_facecolor('#FFFFFF')
    ax1.plot(epochs, history.history['loss'], 'o-', color='#1F77B4', linewidth=2, label='Training Loss')
    ax1.plot(epochs, history.history['val_loss'], 's--', color='#D62728', linewidth=2, label='Validation Loss')
    ax1.set_title(f'Kurva Loss: {title}', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', frameon=True)
    
    # 2. Plot Akurasi
    ax2.set_facecolor('#FFFFFF')
    ax2.plot(epochs, [acc * 100 for acc in history.history['accuracy']], 'o-', color='#2CA02C', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs, [acc * 100 for acc in history.history['val_accuracy']], 's--', color='#FF7F0E', linewidth=2, label='Validation Accuracy')
    ax2.set_title(f'Kurva Akurasi: {title}', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Akurasi (%)', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', frameon=True)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Grafik history disimpan ke: {save_path}")

def plot_confusion_matrix_custom(cm, class_names, title, save_path):
    """
    Membuat visualisasi heatmap Confusion Matrix.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar=True, annot_kws={'size': 13, 'fontweight': 'bold'}
    )
    plt.title(f'Confusion Matrix\n{title}', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Prediksi Model', fontsize=11, fontweight='bold')
    plt.ylabel('Ground Truth (Aktual)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Confusion Matrix disimpan ke: {save_path}")

def plot_all_scenarios_comparison(scenario_results, save_path):
    """
    Membuat grafik batang perbandingan performa akurasi dan F1-Score seluruh skenario.
    """
    names = list(scenario_results.keys())
    accuracies = [scenario_results[k]['test_accuracy'] * 100 for k in names]
    f1_scores = [scenario_results[k]['f1_macro'] * 100 for k in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#FFFFFF')
    
    rects1 = ax.bar(x - width/2, accuracies, width, label='Test Accuracy (%)', color='#2B6CB0', edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x + width/2, f1_scores, width, label='Macro F1-Score (%)', color='#38A169', edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Persentase (%)', fontsize=11, fontweight='bold')
    ax.set_title('Perbandingan Komparatif Performa Model CNN Antar-Skenario', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    
    # Nilai di atas bar
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Perbandingan komparatif disimpan ke: {save_path}")
