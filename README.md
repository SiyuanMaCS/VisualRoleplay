# ⛓‍💥 Visual-RolePlay: Universal Jailbreak Attack on MultiModal Large Language Models via Role-playing Image Character

[Siyuan Ma*](https://github.com/SiyuanMaCS)
<!-- ,[Weidi Luo*](https://github.com/EddyLuo1232), , [Xiaogeng Liu](https://scholar.google.com/citations?user=Gvs5nz8AAAAJ&hl=en&oi=ao), [Xiaoyu Guo](https://github.com/xiaoyuguo6), [Chaowei Xiao†](https://xiaocw11.github.io/) -->

<!-- (*Contribute equally, †Corresponding author) -->

 <!-- [**🛎 Project Page**](https://eddyluo1232.github.io/JailBreakV28K/) |  -->
 
 [**😀 arxiv**](https://arxiv.org/abs/2405.20773)


**<font color='red'>Warning: This repo contains examples of harmful language and images, and reader discretion is recommended.</font>**

## 👻 Installation

```
pip install -r requirements.txt
```


## 📰 News
| Date       | Event    |
|------------|----------|
| **2024/06/12** | 🎁 We have posted our paper on Arxiv.|



## 💡 Abstract
With the advent and widespread deployment of Multimodal Large Language Mod-
els (MLLMs), ensuring their safety has become increasingly critical. To achieve
this objective, it requires us to proactively discover the vulnerabilities of MLLMs
by exploring attack methods. Thus, structure-based jailbreak attacks, where harm-
ful semantic content is embedded within images, have been proposed to mislead
the models. However, previous structure-based jailbreak methods mainly focus on
transforming the format of malicious queries, such as converting harmful content
into images through typography, which lacks sufficient jailbreak effectiveness
and generalizability. To address these limitations, we first introduce the concept
of “Role-play” into MLLM jailbreak attacks and propose a novel and effective method called Visual Role-play (VRP). Specifically, VRP leverages Large Language Models to generate detailed descriptions of high-risk characters and create corresponding images based on the descriptions. When paired with benign role-play
instruction texts, these high-risk character images effectively mislead MLLMs into
generating malicious responses by enacting characters with negative attributes. We
further extend our VRP method into a universal setup to demonstrate its generaliz-
ability. Extensive experiments on popular benchmarks show that VRP outperforms the strongest baselines, Query relevant and FigStep, by an average Attack Success Rate (ASR) margin of 14.3% across all models.
<img src="./assets/vrp.png" width="700"/>
#


## 🛠️ Installation
We take Qwen-VL-Chat and Llava-v1.6-Mistral-7b-hf showcase our attacks. 
### 1. Set up the environment
``
pip install -r requirements.txt
``

### 2. Prepare the weights for Qwen-VL-Chat and Llava-v1.6-Mistral-7b-hf
To access model checkpoints, please first login to huggingface.
```bash
huggingface-cli login
```

## 🚀 Query-specific Visual Role-play
In Query-specific setting, VRP generates characters targeting each malicious query.
### 1. Generate Query-specific Characters
```bash
cd query_specific
bash scripts/generation.sh
```
### 2. Attack Qwen-VL-Chat and Llava-v1.6-Mistral-7b-hf
```bash
bash scripts/attack_qwen.sh
bash scripts/attack_llava.sh
```
### 3. Evaluate
```bash
bash evaluate.sh
```

## 🌏 Universal Visual Role-play
In Universal setting, VRP leverage the optimization capabilities of LLMs to generate candidate characters universally, followed by the selection of the best universal character.
### 1. Generate Candidate Characters
```bash
cd universal
bash scripts/train_qwen.sh
bash scripts/train_llava.sh
```
### 2. Select Best Universal Characters
```bash
bash scripts/valid_qwen.sh
bash scripts/valid_llava.sh
```
### 3. Evaluate
```bash
bash scripts/test_qwen.sh
bash scripts/test_llava.sh
```

## ❌ Disclaimers
This dataset contains offensive content that may be disturbing, This benchmark is provided for educational and research purposes only.

## 📲 Contact
<!-- - Weidi Luo: luo.1455@osu.edu -->
- Siyuan Ma: siyuan.ma.jasper@outlook.com
<!-- - Xiaogeng Liu: xiaogeng.liu@wisc.edu
- Chaowei Xiao: cxiao34@wisc.edu -->

## 📖 BibTeX:
```python
# @misc{ma2024visualroleplay,
#       title={Visual-RolePlay: Universal Jailbreak Attack on MultiModal Large Language Models via Role-playing Image Character}, 
#       author={Siyuan Ma and Weidi Luo and Yu Wang and Xiaogeng Liu and Muhao Chen and Bo Li and Chaowei Xiao},
#       year={2024},
#       eprint={2405.20773},
#       archivePrefix={arXiv},
#       primaryClass={cs.CR}
# }
```