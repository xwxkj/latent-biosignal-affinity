# Latent Biosignal Affinity 实验代码包

## 目标

本代码包用于验证论文中的核心假设：

> 具有相似心理、压力或病理状态的人，在人体多模态信号的潜在空间中应表现出更高的统计相似性。

当前版本优先支持两个最稳的数据集：

1. **PTB-XL**：病理亲和性实验，ECG pathological affinity。
2. **WESAD**：压力/情绪亲和性实验，wearable stress-affinity。

AMIGOS 也保留了接口，但 AMIGOS 通常需要通过官方页面下载预处理 `.mat` 文件，因此代码不强行自动下载。

---

## 推荐先跑哪个？

### 首选：PTB-XL

PTB-XL 是最稳的 proof-of-concept：

- 样本量大；
- ECG 标签强；
- 数据结构清楚；
- 可自动下载；
- 最容易证明 pathological biosignal affinity。

推荐先运行：

```bash
python run_all.py --dataset ptbxl --download --max-records 3000
```

如果机器空间充足，可以提高：

```bash
python run_all.py --dataset ptbxl --download --max-records 8000
```

---

## 安装环境

建议 Python 3.10--3.13。

```bash
cd latent_biosignal_affinity_code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 一键运行 PTB-XL 病理亲和性实验

```bash
python run_all.py --dataset ptbxl --download --max-records 3000
```

输出目录：

```text
results/ptbxl/
├── report.md
├── latent_embedding.csv
├── pairwise_summary.csv
├── fig_embedding.png
├── fig_similarity_boxplot.png
└── fig_permutation_test.png
```

---

## 一键运行 WESAD 压力亲和性实验

```bash
python run_all.py --dataset wesad --download
```

如果自动下载失败，请手动下载 WESAD.zip 并放到：

```text
data/WESAD.zip
```

然后运行：

```bash
python run_all.py --dataset wesad
```

---

## AMIGOS 使用方式

AMIGOS 官方预处理文件通常为：

```text
Data_Preprocessed_P01.mat
...
Data_Preprocessed_P40.mat
```

将它们放到：

```text
data/amigos/data_preprocessed/
```

然后运行：

```bash
python run_all.py --dataset amigos
```

注意：AMIGOS 的 `.mat` 结构可能因下载版本不同而略有差异，因此当前脚本以“尽量兼容”为原则，若遇到字段差异，需要根据文件结构微调 `lba/amigos.py`。

---

## 实验逻辑

### PTB-XL

1. 下载 metadata 和 low-resolution 100 Hz ECG 记录；
2. 从 12 导联 ECG 中提取统计、频域和动态特征；
3. 将特征标准化并用 PCA 得到低维潜在表示；
4. 构建 biosignal similarity matrix；
5. 根据 ECG diagnostic superclass 构建 pathological-affinity matrix；
6. 比较 same-pathology pairs 和 different-pathology pairs 的相似性；
7. 使用 permutation test 检验显著性；
8. 自动生成图表和 Markdown 报告。

### WESAD

1. 读取每个 subject 的 wearable physiological signals；
2. 按 baseline/stress/amusement 状态切分；
3. 对每个 subject-state 提取多模态特征；
4. 构建 latent biosignal space；
5. 检验 same-state pairs 是否比 different-state pairs 更相似；
6. 输出图表和报告。

---

## 论文中建议写法

如果 PTB-XL 实验显著：

> We first tested latent biosignal affinity in a pathological setting using PTB-XL. Pairwise similarity in ECG-derived latent biosignal representations was significantly higher for individuals sharing the same diagnostic superclass than for individuals from different diagnostic superclasses. This result supports the pathological-affinity component of the proposed framework.

如果 WESAD 实验显著：

> We further evaluated stress-related biosignal affinity using WESAD. Subject-state representations derived from wearable physiological signals showed higher similarity within the same affective state than across different states, supporting the stress-affinity prediction of the framework.

---

## 注意事项

- 这不是“证明人体信号决定性格”。
- 实验目标是验证统计亲和性，不是个体级判断。
- 需要在论文中严格写明 confounding factors 和 ethical boundaries。
- 发表时建议补充 permutation test、bootstrap CI、confound regression 和 negative controls。
