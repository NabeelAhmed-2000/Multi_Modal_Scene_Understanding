# Multi-Modal Scene Understanding Pipeline 🖼️🧠

> **Course:** CS 584: Natural Language Processing (Fall 2024)
> **Status:** Complete & Executable

## 👥 Team Members
* **[Your Name]** - [Your Email Address]
* **[Teammate Name]** - [Teammate Email Address]
* **[Teammate Name]** - [Teammate Email Address]

---

## 📖 Overview
This project implements a novel **Multi-Modal Scene Understanding Pipeline** that integrates Object Detection (DETR), Vision-Language Captioning (BLIP-2), and Large Language Model Reasoning (Qwen 2.5).

Unlike standard image captioning which often misses details, our pipeline resolves visual conflicts by "reasoning" over individual object detections.

### System Architecture
![System Pipeline Architecture](assets/pipeline.png)
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
    Execute the notebook cells in order.
    * **Cells 1-2:** Download the COCO validation dataset (approx 60 images).
    * **Phase 1-3:** Perform detection and local captioning.
    * **Phase 4:** Run the Qwen-2.5 Reasoning Agent.
    * **Evaluation:** The final cells generate the performance metrics and plots.

---

## 📊 Evaluation & Results
We evaluated our pipeline against a standard End-to-End baseline using three key metrics:

* **CLIP Score:** Measures how accurately the text describes the image content.
* **BERTScore:** Measures the semantic similarity to human ground truth.
* **Noun Count:** Measures the information density (detail level) of the caption.

### Summary of Results
| Model | CLIP Score (Accuracy) | BERTScore (Human-likeness) | Noun Count (Detail) |
| :--- | :--- | :--- | :--- |
| **Baseline (End-to-End)** | [Insert Value from Notebook] | [Insert Value] | [Insert Value] |
| **Ours (Qwen-2.5)** | **[Insert Value]** | **[Insert Value]** | **[Insert Value]** |

> *Note: Our pipeline achieves significantly higher detail (Noun Count) while maintaining competitive semantic accuracy.*

---

## 🛠️ Tech Stack
* **Object Detection:** DETR (ResNet-50)
* **Captioning:** BLIP-2 (Opt-2.7b)
* **Reasoning Agent:** Qwen-2.5-7B-Instruct (4-bit Quantized)
* **Infrastructure:** PyTorch, HuggingFace Transformers, Google Colab (A100)

## 📂 Files
* `Multi_Modal_Scene_Understanding.ipynb`: The core research implementation.
* `requirements.txt`: Python dependency list.
* `assets/`: Supplementary visualizations.
