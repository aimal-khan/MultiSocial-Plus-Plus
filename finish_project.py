# ============================================================
# MultiSocial++ — Local Recovery Script
# Run this from your project folder in VS Code
# ============================================================
import sys, os, json, time, pickle, re, warnings
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # no display needed
warnings.filterwarnings('ignore')

from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset as HFDataset
from transformers import DataCollatorWithPadding

# ── EDIT THIS PATH to wherever you saved your outputs folder ──
BASE = r"C:\Users\Aimalicious\Desktop\THIS SEM\NLPDC Project\outputs"
# BASE = "/home/yourname/MultiSocial_Project/outputs"  # Linux/Mac
# ─────────────────────────────────────────────────────────────

DEVICE = torch.device('cpu')  # CPU only locally
SEED   = 42
MAX_SEQ = 128
np.random.seed(SEED)
torch.manual_seed(SEED)

print("="*55)
print("  MultiSocial++ — Local Recovery Script")
print("="*55)
print(f"  Device : {DEVICE}")
print(f"  Base   : {BASE}")
print()

# ── Step 1: Load preprocessed cache ──────────────────────────
print("📂 Loading preprocessed cache...")
cache_path = os.path.join(BASE, "preprocessed_cache.pkl")
df = pd.read_pickle(cache_path)
print(f"✅ Loaded {len(df):,} rows")
print(f"   Columns: {list(df.columns)}")

# ── Step 2: Identify columns ──────────────────────────────────
# Auto-detect columns
TEXT_COL  = 'clean_text'
LABEL_COL = 'binary_label'

# Find language column
for c in df.columns:
    if 'lang' in c.lower():
        LANG_COL = c; break
else:
    LANG_COL = None

# Find platform column
for c in df.columns:
    if 'source' in c.lower() or 'platform' in c.lower():
        PLAT_COL = c; break
else:
    PLAT_COL = None

print(f"   Text col     : {TEXT_COL}")
print(f"   Label col    : {LABEL_COL}")
print(f"   Language col : {LANG_COL}")
print(f"   Platform col : {PLAT_COL}")

# ── Step 3: Create splits ─────────────────────────────────────
texts  = df[TEXT_COL].tolist()
labels = df[LABEL_COL].tolist()
langs  = df[LANG_COL].tolist()  if LANG_COL else [None]*len(df)
plats  = df[PLAT_COL].tolist()  if PLAT_COL else [None]*len(df)

X_train,X_temp,y_train,y_temp,l_train,l_temp,p_train,p_temp = train_test_split(
    texts,labels,langs,plats,test_size=0.30,random_state=SEED,stratify=labels)
X_val,X_test,y_val,y_test,l_val,l_test,p_val,p_test = train_test_split(
    X_temp,y_temp,l_temp,p_temp,test_size=0.50,random_state=SEED,stratify=y_temp)

print(f"\n✅ Splits: Train={len(X_train):,} Val={len(X_val):,} Test={len(X_test):,}")

# ── Step 4: TF-IDF + LR (fast, runs in seconds) ──────────────
print("\n🔤 Training TF-IDF + LR...")
t0 = time.time()
vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf=True)
Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)
lr  = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED, n_jobs=-1)
lr.fit(Xtr, y_train)
t_lr = time.time()-t0
y_pred_lr = lr.predict(Xte)
f1_lr  = f1_score(y_test, y_pred_lr, average='macro')
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"✅ TF-IDF+LR → Macro-F1: {f1_lr:.4f} | Acc: {acc_lr:.4f} | Time: {t_lr:.1f}s")

# ── Step 5: Evaluate saved XLM-R models ──────────────────────
# Use small subset for CPU speed (500 samples ~5 mins on CPU)
EVAL_N = 500
X_eval  = X_test[:EVAL_N]
y_eval  = y_test[:EVAL_N]
l_eval  = l_test[:EVAL_N]
p_eval  = p_test[:EVAL_N]

def evaluate_model(model_path, label):
    print(f"\n📊 Evaluating: {label}")
    print(f"   Loading from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    preds, times = [], []
    BATCH = 16
    for i in range(0, len(X_eval), BATCH):
        batch = X_eval[i:i+BATCH]
        enc = tokenizer(batch, truncation=True, max_length=MAX_SEQ,
                        padding=True, return_tensors='pt')
        t0 = time.time()
        with torch.no_grad():
            out = model(**enc)
        times.append(time.time()-t0)
        preds.extend(torch.argmax(out.logits, dim=-1).tolist())
        if i % 100 == 0:
            print(f"   ... {i}/{len(X_eval)}")
    f1  = f1_score(y_eval, preds, average='macro')
    acc = accuracy_score(y_eval, preds)
    throughput = len(X_eval) / sum(times)
    print(f"✅ {label} → Macro-F1: {f1:.4f} | Acc: {acc:.4f} | Throughput: {throughput:.0f} texts/s")
    return f1, acc, throughput, preds, tokenizer, model

baseline_path  = os.path.join(BASE, "xlmr_baseline")
optimized_path = os.path.join(BASE, "xlmr_optimized")

f1_base, acc_base, thr_base, preds_base, _, _ = evaluate_model(baseline_path, "XLM-R Baseline")
f1_opt,  acc_opt,  thr_opt,  preds_opt, tokenizer, model_opt = evaluate_model(optimized_path, "XLM-R Optimized")

# ── Step 6: ONNX Export ───────────────────────────────────────
print("\n📦 Exporting to ONNX...")
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

onnx_path = os.path.join(BASE, "xlmr_onnx")
quant_path = os.path.join(BASE, "xlmr_quantized")

t0 = time.time()
ort_model = ORTModelForSequenceClassification.from_pretrained(optimized_path, export=True)
ort_model.save_pretrained(onnx_path)
tokenizer.save_pretrained(onnx_path)
print(f"✅ ONNX exported in {time.time()-t0:.1f}s")

# ── Step 7: INT8 Quantization ─────────────────────────────────
print("\n🔢 Quantizing to INT8...")
t0 = time.time()
quantizer = ORTQuantizer.from_pretrained(onnx_path)
qconfig   = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
quantizer.quantize(save_dir=quant_path, quantization_config=qconfig)
print(f"✅ Quantized in {time.time()-t0:.1f}s")

orig_mb  = os.path.getsize(os.path.join(onnx_path, "model.onnx")) / 1e6
quant_mb = os.path.getsize(os.path.join(quant_path, "model_quantized.onnx")) / 1e6
print(f"   ONNX size     : {orig_mb:.1f} MB")
print(f"   Quantized size: {quant_mb:.1f} MB ({100*(1-quant_mb/orig_mb):.0f}% smaller)")

# ── Step 8: ONNX Inference Benchmark ─────────────────────────
print("\n⚡ Benchmarking ONNX inference...")

def onnx_throughput(ort_m, texts, batch=16):
    total_time = 0
    for i in range(0, len(texts), batch):
        b = texts[i:i+batch]
        enc = tokenizer(b, truncation=True, max_length=MAX_SEQ,
                        padding=True, return_tensors='pt')
        t0 = time.time()
        out = ort_m(**enc)
        total_time += time.time()-t0
    return len(texts)/total_time

ort_infer = ORTModelForSequenceClassification.from_pretrained(onnx_path)
ort_quant = ORTModelForSequenceClassification.from_pretrained(
    quant_path, file_name="model_quantized.onnx")

thr_onnx  = onnx_throughput(ort_infer, X_eval)
thr_quant = onnx_throughput(ort_quant,  X_eval)

# F1 for ONNX
def onnx_preds(ort_m, texts, batch=16):
    all_p = []
    for i in range(0, len(texts), batch):
        b = texts[i:i+batch]
        enc = tokenizer(b, truncation=True, max_length=MAX_SEQ, padding=True, return_tensors='pt')
        out = ort_m(**enc)
        all_p.extend(out.logits.argmax(dim=-1).tolist())
    return all_p

f1_onnx  = f1_score(y_eval, onnx_preds(ort_infer, X_eval), average='macro')
f1_quant = f1_score(y_eval, onnx_preds(ort_quant,  X_eval), average='macro')

print(f"   PyTorch  : {thr_base:.0f} texts/s | F1: {f1_base:.4f}")
print(f"   ONNX     : {thr_onnx:.0f} texts/s | F1: {f1_onnx:.4f}  ({thr_onnx/thr_base:.2f}x)")
print(f"   ONNX+INT8: {thr_quant:.0f} texts/s | F1: {f1_quant:.4f}  ({thr_quant/thr_base:.2f}x)")

# ── Step 9: Per-language analysis ─────────────────────────────
print("\n📊 Per-language F1 (Optimized XLM-R):")
lang_f1 = {}
for lang in set(l_eval):
    idx = [i for i,l in enumerate(l_eval) if l == lang]
    if len(idx) < 5: continue
    f1_l = f1_score([y_eval[i] for i in idx], [preds_opt[i] for i in idx],
                    average='macro', zero_division=0)
    lang_f1[lang] = round(f1_l, 4)

for lang, f1 in sorted(lang_f1.items(), key=lambda x: x[1], reverse=True):
    print(f"   {lang:6s}: {f1:.4f}")

# ── Step 10: Save final results ───────────────────────────────
results = {
    "preprocessing": {
        "sequential_sec": 1.706,
        "parallel_sec": 1.389,
        "speedup": 1.228,
        "texts_processed": 50000,
        "cpu_cores": 2
    },
    "tfidf_lr":      {"macro_f1": round(f1_lr,4),   "accuracy": round(acc_lr,4),  "train_time_sec": round(t_lr,2)},
    "xlmr_baseline": {"macro_f1": round(f1_base,4),  "accuracy": round(acc_base,4)},
    "xlmr_optimized":{"macro_f1": round(f1_opt,4),   "accuracy": round(acc_opt,4)},
    "inference": {
        "pytorch":   {"throughput": round(thr_base),  "macro_f1": round(f1_base,4)},
        "onnx":      {"throughput": round(thr_onnx),  "macro_f1": round(f1_onnx,4),  "speedup": round(thr_onnx/thr_base,3)},
        "onnx_int8": {"throughput": round(thr_quant), "macro_f1": round(f1_quant,4), "speedup": round(thr_quant/thr_base,3)},
    },
    "model_sizes": {"onnx_mb": round(orig_mb,1), "quantized_mb": round(quant_mb,1)},
    "per_language_f1": lang_f1
}

with open(os.path.join(BASE, "benchmark_results_FINAL.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\n💾 Saved benchmark_results_FINAL.json")

# ── Step 11: Generate all plots ───────────────────────────────
print("\n🎨 Generating plots...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('MultiSocial++ — Results Summary', fontsize=14, fontweight='bold')

# F1 comparison
methods = ['TF-IDF\n+LR','XLM-R\nBaseline','XLM-R\nOptimized','ONNX','ONNX\nINT8']
f1s     = [f1_lr, f1_base, f1_opt, f1_onnx, f1_quant]
colors  = ['#90A4AE','#42A5F5','#66BB6A','#FFA726','#EF5350']
bars = axes[0].bar(methods, f1s, color=colors, edgecolor='black')
axes[0].set_title('Macro-F1 Score'); axes[0].set_ylim(0,1)
for b,v in zip(bars,f1s):
    axes[0].text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

# Inference throughput
inf_methods = ['PyTorch','ONNX','ONNX+INT8']
throughputs  = [thr_base, thr_onnx, thr_quant]
axes[1].bar(inf_methods, throughputs, color=['#42A5F5','#FFA726','#66BB6A'], edgecolor='black')
axes[1].set_title('Inference Throughput (texts/sec)')
for i,t in enumerate(throughputs):
    axes[1].text(i, t+1, f'{t:.0f}', ha='center', fontweight='bold')

# Per-language F1
if lang_f1:
    sorted_langs = sorted(lang_f1.items(), key=lambda x: x[1], reverse=True)[:10]
    axes[2].barh([x[0] for x in sorted_langs], [x[1] for x in sorted_langs], color='#9C27B0', edgecolor='black')
    axes[2].set_title('Per-Language Macro-F1')
    axes[2].set_xlim(0,1)

plt.tight_layout()
plt.savefig(os.path.join(BASE, "results_summary.png"), dpi=150)
print(f"✅ Saved results_summary.png")

# Ablation table
print("\n" + "="*65)
print("ABLATION STUDY — MultiSocial++")
print("="*65)
print(f"  {'Method':<30} {'Macro-F1':>10} {'Inf Speed':>12}")
print(f"  {'-'*52}")
print(f"  {'TF-IDF + LR':<30} {f1_lr:>10.4f} {'N/A':>12}")
print(f"  {'XLM-R Baseline':<30} {f1_base:>10.4f} {thr_base:>10.0f}/s")
print(f"  {'XLM-R + All PDC Opts':<30} {f1_opt:>10.4f} {thr_opt:>10.0f}/s")
print(f"  {'+ ONNX Export':<30} {f1_onnx:>10.4f} {thr_onnx:>10.0f}/s")
print(f"  {'+ ONNX + INT8 (Full++)':<30} {f1_quant:>10.4f} {thr_quant:>10.0f}/s")
print("="*65)
print(f"\n🚀 Inference speedup (INT8 vs PyTorch): {thr_quant/thr_base:.2f}x")
print(f"📉 F1 drop (INT8 vs Optimized)        : {f1_opt-f1_quant:.4f}")
print("\n✅ ALL DONE! Check your outputs folder.")