# MultiSocial++: Efficient Multilingual AI-Text Detection

MultiSocial++ evaluates and optimizes transformer models for detecting AI-generated text across 18 languages. Using Parallel and Distributed Computing (PDC) techniques, the pipeline reduces inference latency and memory footprints for deployment on CPU environments.

## Highlights
- Multilingual evaluation across 18 languages including English, Urdu, Polish, Portuguese, and Catalan.
- Parallel preprocessing pipeline for cleaning social media corpora.
- Mixed-precision (FP16) training and parallel data loading.
- ONNX export and INT8 quantization for CPU deployment.

## Performance Benchmark

| Metric | PyTorch Baseline | ONNX + INT8 (Optimized) | Improvement |
|:---|:---|:---|:---|
| Inference Throughput | 5 texts/sec | 11 texts/sec | 1.93x speedup |
| Model Size | 1,112 MB | 279 MB | ~75% smaller |
| Macro-F1 Score | 0.7632 | 0.7639 | Maintained accuracy |

### Per-Language F1 (Top 5)
- Polish: 0.8906
- Portuguese: 0.8857
- English: 0.8825
- Catalan: 0.8523
- German: 0.8182

## Repository Structure
- `MultiSocial_Complete_Project.ipynb`: Training, exploratory data analysis, and optimization pipeline.
- `finish_project.py`: Local benchmarking and inference demonstration script.
- `multisocial_anonymized.csv.gz`: Multilingual social media dataset.
- `outputs/`: Evaluation logs, ablation metrics, and summary figures.

## Setup & Benchmark

```bash
pip install transformers datasets onnxruntime optimum pandas scikit-learn
python finish_project.py
```

---

Developed by Aimal Khan and Awais Arshad.
