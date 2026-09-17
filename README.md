# TUGAS UJIAN TENGAH SEMESTER (UTS) — DEEP LEARNING (KELAS A)
## Eksperimen Optimasi Bertingkat (Progressive Ablation Study): Arsitektur Convolutional Neural Network (CNN) pada Citra CT-Scan Thoraks Kanker Paru-Paru (IQ-OTH/NCCD)

[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset: Mendeley Data](https://img.shields.io/badge/Mendeley_Data-10.17632%2Fbhmdr45bh2.2-blue)](https://doi.org/10.17632/bhmdr45bh2.2)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/attaramadhani/TUGAS-UTS_DEEP-LEARNING-A/blob/main/Tugas_UTS_CNN_IQOTHNCCD.ipynb)

---

### 👥 Identitas Anggota Kelompok 6:
| No. | Nama Lengkap | NIM | Kelas | Program Studi |
| :---: | :--- | :---: | :---: | :---: |
| 1. | **Attala Alif Ramadhani Tri Hida** | `230441100144` (23-144) | Deep Learning (A) | Sistem Informasi |
| 2. | **Nafaul Hernanda Romadlona** | `240441100125` (24-125) | Deep Learning (A) | Sistem Informasi |
| 3. | **M.Rafly Kurniawan** | `240441100086` (24-086) | Deep Learning (A) | Sistem Informasi |
| 4. | **Naufal Husain** | `240441100038` (24-038) | Deep Learning (A) | Sistem Informasi |

* **Dosen Pengampu:** Dr. Wahyudi Setiawan, S.Kom., M.Kom.  
* **Program Studi:** S1 Sistem Informasi, Fakultas Teknik, Universitas Trunojoyo Madura  

---

## 📌 Ringkasan Proyek
Repository ini berisi implementasi lengkap tugas Ujian Tengah Semester (UTS) mata kuliah Deep Learning untuk mengklasifikasikan citra CT-Scan rongga dada (kanker paru-paru) ke dalam 3 kelas:
1. **Benign cases** (Tumor Jinak)
2. **Malignant cases** (Kanker Ganas)
3. **Normal cases** (Jaringan Sehat)

Eksperimen dirancang dengan paradigma **Optimasi Bertingkat (Progressive Ablation Study)**: 1 arsitektur dasar CNN yang sama dievaluasi melalui 4 skenario optimasi berantai. Konfigurasi terbaik dari tiap skenario diwariskan sebagai fondasi pengujian pada skenario berikutnya hingga terpilih model akhir (**Final Champion Model**).

---

## 🗂️ Informasi Dataset Sumber Terpercaya
* **Sumber Data:** [Mendeley Data — The IQ-OTHNCCD Lung Cancer Dataset](https://doi.org/10.17632/bhmdr45bh2.2)
* **Penulis Dataset:** Hamdalla Alyasriy & Muayed AL-Huseiny (Wasit University & IQ-OTH/NCCD Oncology Centers, Irak)
* **DOI:** `10.17632/bhmdr45bh2.2` (Lisensi CC BY 4.0)
* **Total Citra:** **1.097 citra** format DICOM/PNG/JPEG beresolusi asli 512×512 piksel:
  - **Benign:** 120 citra (10.94%)
  - **Malignant:** 561 citra (51.14%)
  - **Normal:** 416 citra (37.92%)

---

## 🧱 Arsitektur Model CNN (Custom 4-Block)
Model yang digunakan adalah arsitektur *Deep Convolutional Neural Network* modular 4 blok:
- **Input Layer:** `(128, 128, 3)` ternormalisasi $[0.0, 1.0]$.
- **Blok Konvolusi 1:** `Conv2D(32, kernel 3x3, padding='same', activation='relu')` $\rightarrow$ `MaxPooling2D(2x2)`
- **Blok Konvolusi 2:** `Conv2D(64, kernel 3x3, padding='same', activation='relu')` $\rightarrow$ `MaxPooling2D(2x2)`
- **Blok Konvolusi 3:** `Conv2D(128, kernel 3x3, padding='same', activation='relu')` $\rightarrow$ `MaxPooling2D(2x2)`
- **Blok Konvolusi 4:** `Conv2D(128, kernel 3x3, padding='same', activation='relu')` $\rightarrow$ `MaxPooling2D(2x2)`
- **Dense Classifier:** `Flatten()` $\rightarrow$ `Dense(128, activation='relu')` $\rightarrow$ `Dropout(rate=p)` $\rightarrow$ `Dense(3, activation='softmax')`

---

## 🔬 Metodologi & Hasil Eksperimen 4 Skenario Bertingkat

```
[Skenario 1: Variasi Data Split] ──> Pilih Split 90:05:05 (Akurasi 94.55%)
                 │
                 ▼
[Skenario 2: Uji Augmentasi Data] ──> Pilih Tanpa Augmentasi (Akurasi 94.55%)
                 │
                 ▼
[Skenario 3: Komparasi Optimizer] ──> Pilih Adam (Akurasi 94.55%)
                 │
                 ▼
[Skenario 4: Studi Dropout]       ──> Pilih Dropout 0.3 (Akurasi 98.18%, F1 96.62%)
                 │
                 ▼
     [FINAL CHAMPION MODEL]
```

### Rekapitulasi Metrik Evaluasi:
| Skenario | Varian yang Diuji | Akurasi Uji | Loss Uji | Macro Precision | Macro Recall | Macro F1-Score | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Skenario 1** | Split 70:15:15 | 93.33% | 0.2038 | 91.56% | 90.96% | 91.24% | - |
| **Skenario 1** | Split 80:10:10 | 92.73% | 0.2520 | 89.92% | 89.28% | 89.47% | - |
| **Skenario 1** | **Split 90:05:05** | **94.55%** | **0.1713** | **92.20%** | **91.13%** | **91.56%** | **Juara S1** 🏆 |
| **Skenario 2** | **Tanpa Augmentasi** | **94.55%** | **0.1713** | **92.20%** | **91.13%** | **91.56%** | **Juara S2** 🏆 |
| **Skenario 2** | Dengan Augmentasi | 89.09% | 0.2974 | 88.35% | 85.08% | 86.20% | - |
| **Skenario 3** | **Adam (lr=0.0005)** | **94.55%** | **0.1713** | **92.20%** | **91.13%** | **91.56%** | **Juara S3** 🏆 |
| **Skenario 3** | RMSprop (lr=0.0005) | 90.91% | 0.3804 | 93.33% | 81.39% | 85.20% | - |
| **Skenario 3** | SGD Momentum | 83.64% | 0.3951 | 80.82% | 75.28% | 76.87% | - |
| **Skenario 4** | Dropout 0.0 | 94.55% | 0.2317 | 93.59% | 92.88% | 93.07% | - |
| **Skenario 4** | **Dropout 0.3** | **98.18%** | **0.1066** | **96.67%** | **96.67%** | **96.62%** | **CHAMPION MODEL** 🌟 |
| **Skenario 4** | Dropout 0.5 | 96.36% | 0.1723 | 94.87% | 95.83% | 95.28% | - |

---

## 📁 Struktur File Repositori

```text
TUGAS-UTS_DEEP-LEARNING-A/
│
├── .gitignore                                  # Mengabaikan file dataset besar & binary model
├── README.md                                   # Dokumentasi lengkap & ringkasan hasil
├── download_and_verify.py                      # Skrip pengunduh otomatis dataset resmi
├── run_progressive_pipeline.py                 # Pipeline eksekusi 4 skenario bertingkat
├── generate_report_docx.py                     # Skrip pembuat dokumen laporan Word formal
│
├── Tugas_UTS_CNN_IQOTHNCCD.ipynb               # Jupyter Notebook lengkap dengan visualisasi output
├── Laporan_Lengkap_UTS_DeepLearning_CNN.docx   # Dokumen laporan resmi UTS berformat Word
│
├── src/                                        # Modul Python fungsional
│   ├── __init__.py
│   ├── data_preprocessing.py                   # Pemuatan, normalisasi, & augmentasi citra
│   ├── model_builder.py                        # Arsitektur Conv-ReLU-Pool CNN & kompilasi
│   └── evaluation_utils.py                     # Metrik evaluasi, kurva pelatihan, confusion matrix
│
└── outputs/                                    # Artefak visualisasi & catatan log
    ├── figures/                                # Grafik komparasi, confusion matrix, & ROC (300 DPI)
    └── logs/                                   # Rekapitulasi riwayat pelatihan CSV & JSON
```

---

## 🚀 Panduan Menjalankan Kode Program

### 1. Klon Repositori & Persiapan Lingkungan
```bash
git clone https://github.com/attaramadhani/TUGAS-UTS_DEEP-LEARNING-A.git
cd TUGAS-UTS_DEEP-LEARNING-A
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn pillow kagglehub python-docx
```

### 2. Unduh Dataset Secara Otomatis
```bash
python download_and_verify.py
```

### 3. Jalankan Eksperimen Optimasi Bertingkat (Progressive Pipeline)
```bash
python run_progressive_pipeline.py
```

### 4. Buat Dokumen Laporan Word (.docx) Lengkap
```bash
python generate_report_docx.py
```

### 5. Membuka Jupyter Notebook
Buka `Tugas_UTS_CNN_IQOTHNCCD.ipynb` menggunakan VS Code, Jupyter Lab, atau unggah langsung ke **Google Colab**. Seluruh blok kode telah tereksekusi rapi dan menyajikan kurva pelatihan serta matriks evaluasi interaktif.

---

## 📚 Referensi Dataset
Alyasriy, H., & AL-Huseiny, M. (2020). *The IQ-OTHNCCD Lung Cancer Dataset*. Mendeley Data, V2, doi: [10.17632/bhmdr45bh2.2](https://doi.org/10.17632/bhmdr45bh2.2).
