# Multi-Modal Scene Understanding Pipeline 🖼️🧠

## 👥 Team Members
* **[Nabeel Ahmed]** - [nahmed48@gmu.edu]
* **[Samruddhi Pradeep Deshmukh]** - [sdeshmu@gmu.edu]
* **[Sina Mansouri]** - [smansou3@gmu.edu]

---

## 📖 Overview
This project implements a novel **Multi-Modal Scene Understanding Pipeline** that integrates Object Detection (DETR), Vision-Language Captioning (BLIP-2), and Large Language Model Reasoning (Qwen-2.5 & Llama-3).

Unlike standard image captioning which often misses details, our pipeline resolves visual conflicts by "reasoning" over individual object detections.

### System Architecture
![System Pipeline Architecture](assets/pipeline_architecture.png)
*Figure 1: The end-to-end workflow from raw image to final reasoning-based description.*

---

## 🚀 How to Reproduce (Executability)
This project is submitted as a **single, self-contained Jupyter Notebook** to ensure perfect reproducibility of the results and visualizations.

1.  **Open the Notebook:**
    [`Multi_Modal_Scene_Understanding.ipynb`](./Multi_Modal_Scene_Understanding.ipynb)
    *(Recommended: Open in Google Colab using an A100 GPU)*

2.  **Install Dependencies:**
    All necessary libraries are installed via the first cell of the notebook. Alternatively, you can install them locally using:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Pipeline:**
    The pipeline moves through distinct phases:
    * **Phase 1 (Detection):** Utilizing `DETR-ResNet-50` to identify objects.
    * **Phase 2 (Patching):** Dynamic cropping and upscaling of object regions.
    * **Phase 3 (Local Captioning):** Using `BLIP-2` for detailed object-level visual descriptions.
    * **Phase 4 (Reasoning):** Aggregating insights using **Qwen-2.5** and **Llama-3** to synthesize global descriptions.
    * **Phase 5 (Tournament):** Using Llama-3 as an impartial judge to compare pipeline outputs against the baseline.

---

## 📊 Evaluation & Results
We evaluated our pipeline against a standard End-to-End baseline using three key metrics. We tested two different Reasoning Agents: **Qwen-2.5** and **Llama-3**.

* **CLIP Score:** Measures how accurately the text describes the image content.
* **BERTScore:** Measures the semantic similarity to human ground truth.
* **Noun Count:** Measures the information density (detail level) of the caption.

### Summary of Results
| Model | CLIP Score (Accuracy) | BERTScore (Human-likeness) | Noun Count (Detail) |
| :--- | :--- | :--- | :--- |
| **Baseline (End-to-End)** | 28.12 | 0.87 | 12.07 |
| **Ours (Qwen-2.5)** | 27.40 | 0.87 | 25.92 |
| **Ours (Llama-3)** | 22.85 | 0.83 | 54.87 |

### 🏆 LLM-as-a-Judge Tournament
We also implemented an **LLM-based Tournament** (Phase 17) where Llama-3 acted as a judge to blindly compare descriptions from the Baseline vs. Our Pipeline.
* **Llama-3 vs Baseline:** 59-1
* **Qwen-2.5 vs Baseline:** 62-1

> *Note: Both Qwen-2.5 and Llama-3 based pipelines significantly outperformed the baseline in detail richness.*

---

## 🛠️ Tech Stack
* **Object Detection:** DETR (ResNet-50)
* **Captioning:** BLIP-2 (Opt-2.7b)
* **Reasoning Agents:**
  * Qwen-2.5-7B-Instruct (4-bit Quantized)
  * Llama-3-8B-Instruct (4-bit Quantized)
* **Evaluation:** OpenAI CLIP, BERTScore, and LLM-as-a-Judge (Llama-3)
* **Infrastructure:** PyTorch, HuggingFace Transformers, Google Colab (A100)

## 📂 Files
* `Multi_Modal_Scene_Understanding.ipynb`: The core research implementation.
* `requirements.txt`: Python dependency list.
* `assets/`: Supplementary visualizations.
