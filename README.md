# 🚀 MultiSocial++: Efficient Multilingual AI-Text Detection

**MultiSocial++** is an advanced NLP research project focused on detecting AI-generated text across multiple languages with high efficiency. By integrating **Parallel & Distributed Computing (PDC)** techniques, we optimize both the training pipeline and real-time inference latency.

## 🌟 Key Features
- **Multilingual Support**: High-accuracy detection across 18+ languages (English, Urdu, Chinese, etc.).
- **PDC Optimizations**:
  - **Parallel Preprocessing**: Multiprocessing pipeline for cleaning massive social media datasets.
  - **Efficient Training**: Mixed-precision (FP16) and parallel data loading.
  - **Inference Acceleration**: ONNX export and **INT8 Quantization** for CPU deployment.

## 📊 Performance Benchmark

Our optimization strategies achieved significant improvements in speed and efficiency without sacrificing research-grade accuracy.

| Metric | PyTorch (Baseline) | ONNX + INT8 (Optimized) | Improvement |
| :--- | :--- | :--- | :--- |
| **Inference Throughput** | 5 texts/sec | **11 texts/sec** | **1.93x Speedup** |
| **Model Size** | 1,112 MB | **279 MB** | **~75% Smaller** |
| **Macro-F1 Score** | 0.7632 | 0.7639 | Stable Accuracy |

### Per-Language F1 (Top 5)
- **English**: 0.8825
- **Portuguese**: 0.8857
- **Polish**: 0.8906
- **Catalan**: 0.8523
- **German**: 0.8182

## 📁 Repository Structure
- `MultiSocial_Complete_Project.ipynb`: Full training, EDA, and optimization pipeline.
- `finish_project.py`: Local benchmarking and demo script.
- `multisocial_anonymized.csv.gz`: Multilingual social media dataset.
- `outputs/`: Final results, ablation study JSON, and summary plots.

## 🛠️ Setup & Usage
1. Install dependencies:
   ```bash
   pip install transformers datasets onnxruntime optimum pandas scikit-learn
   ```
2. Run the benchmark:
   ```bash
   python finish_project.py
   ```

---
**Project by:** Aimal Khan & Awais Arshad
*Developed as part of the PDC & NLP Research Project Curriculum.*
