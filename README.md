# CRU-Reason: Knowledge Graph-Grounded and Interpretable Clinical Prediction with Clinical Rationale Units

CRU-Reason is a comprehensive framework for clinical reasoning and prediction tasks using electronic health records (EHR).
It supports multiple healthcare prediction tasks including drug recommendation and diagnosis prediction on MIMIC-III and MIMIC-IV datasets.

## 📁 Project Structure

### Root Directory Files

- **`zeroshot.py`** - Zero-shot evaluation script for evaluating models without training (e.g., LLM-based methods).
- **`train.py`** - Main training script for supervised learning. Supports multiple runs with different random seeds and saves metrics with mean/std statistics.
- **`eval.py`** - Evaluation script for testing trained models. Loads checkpoints from saved experiments and evaluates on test sets.

### Core Directories

#### 📂 `data/` - Data Loading and Processing

- **`dataloader.py`** - Main data loading module. Handles dataset loading, patient-level splitting, task-specific processing, and dataloader creation.

#### 📂 `graph/` - Knowledge Graph
Manages medical knowledge graph construction, embedding, and clustering. See **`graph/README.md`**.

#### 📂 `baselines/` - Baseline Methods
Implementation of various baseline methods for comparison.

#### 📂 `TherapyReason/` - our CRU-Reason framework

- **`diagnosis.ipynb`** - Jupyter notebook for diagnosis prediction task.
- **`drugrec.ipynb`** - Jupyter notebook for drug recommendation task.

- **`select.ipynb`** - Selection and filtering notebook.

- **`cluster_distribution.py`** - Cluster distribution analysis.

- **`ablation/`** - Ablation experiments.
- **`backbone/`** - Backbone ablation experiments.
- **`interpret/`** - Interpretability analysis experiments.
- **`ood/`** - Out-of-distribution experiments.
- **`analysis/`** - Analysis utilities (cost analysis, usage statistics, solution analysis).


#### 📂 `finetune/` - Fine-tuning Scripts
Scripts for fine-tuning large language models on medical tasks.

- **`finetune_dataset.py`** - Dataset preparation for fine-tuning.
- **`finetune_run.py`** - Main fine-tuning execution script.
- **`finetune_eval.py`** - Evaluation script for fine-tuned models.
- **`CoT/`** - CoT-specific fine-tuning scripts.
- **`KARE/`** - KARE-specific fine-tuning scripts.
