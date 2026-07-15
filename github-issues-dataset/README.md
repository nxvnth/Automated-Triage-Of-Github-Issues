---
license: mit
language:
- en
size_categories:
- 100K<n<1M
task_categories:
- text-classification
- text-retrieval
tags:
- github
- issues
- bug-tracking
- multi-label
- priority-classification
- severity-classification
pretty_name: GitHub Issues Dataset
---

# 📌 GitHub Issues Dataset
[![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/github-issues-dataset)

📂 **Dataset Name**: `github-issues-dataset`  
📊 **Total Issues**: **114073**  
📜 **Format**: **Parquet (`.parquet`)**  
🔍 **Source**: **GitHub Repositories (Top 100 Repos)**  

---

## 📖 Overview
This dataset contains **114,073 GitHub issues** collected from the **top 100 repositories** on GitHub.  
It is designed for **issue classification, severity/priority prediction, and AI/ML training**.

### ✅ This dataset is useful for:
- **AI/ML Training**: Fine-tune models for issue classification & prioritization.
- **Natural Language Processing (NLP)**: Analyze software development discussions.
- **Bug Severity Prediction**: Train models to classify issues as **Critical, Major, or Minor**.

---

## 📂 Dataset Structure
The dataset is stored in **Parquet format (`github_issues_dataset.parquet`)** for **efficient storage and fast retrieval**.

### **Columns in the Dataset:**

| Column        | Type   | Description |
|--------------|--------|-------------|
| `id`         | `int`  | Github issue id |
| `repo`       | `str`  | Repository name |
| `title`      | `str`  | Issue title |
| `body`       | `str`  | Issue description |
| `labels`     | `list` | Assigned GitHub labels |
| `priority`   | `str`  | Estimated priority (`high`, `medium`, `low`) |
| `severity`   | `str`  | Estimated severity (`Critical`, `Major`, `Minor`) |

---

## 📥 Download & Use

### Using `datasets` Library
You can easily load this dataset using Hugging Face's `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset("sharjeelyunus/github-issues-dataset")
```

---

## 📊 Sample Data

| id         | repo                | title                         | labels        | priority | severity |
|------------|---------------------|-------------------------------|---------------|----------|----------|
| 101        | `pytorch/pytorch`    | "RuntimeError: CUDA out of memory" | `["bug", "cuda"]` | high | Critical |
| 102        | `tensorflow/tensorflow` | "Performance degradation in v2.9" | `["performance"]` | medium | Major |
| 103        | `microsoft/vscode`   | "UI freeze when opening large files" | `["ui", "bug"]` | low | Minor |

---

## 🛠 How This Dataset Was Created
1. **Collected open issues** from the **top 100 repositories** on GitHub.
2. **Filtered only English issues** with **assigned labels**.
3. **Processed priority and severity**:
   - Used **labels** to determine **priority & severity**.
   - Used **ML models** to predict missing priority/severity values.
4. **Stored dataset in Parquet format** for **ML processing**.

---

## 🔍 Use Cases
- **AI-Powered Bug Triage**: Train AI models to predict **priority & severity**.
- **NLP Research**: Analyze software engineering discussions.

---

## 📜 License
This dataset is open-source and **publicly available** under the **MIT License**.  
Please cite this dataset if you use it in research.

---

## 📫 Feedback & Contributions
- Found an issue? Open an **[issue](https://github.com/sharjeelyunus/github-issues-analyzer/issues)**.
- Want to contribute? Feel free to **submit a PR**.
- For any questions, reach out on **[Hugging Face Discussions](https://huggingface.co/datasets/sharjeelyunus/github-issues-dataset/discussions)**.

---

## ⭐ Support
📌 **If you find this dataset useful, please like ❤️ the repository!**  
🚀 **Happy Coding!** 🚀
