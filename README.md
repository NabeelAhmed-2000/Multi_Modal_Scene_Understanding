# Multi-Modal Scene Understanding Pipeline 🖼️🧠

> **Project Status:** Complete & Executable
> **Course:** CS 584: Natural Language Processing (Fall 2024)

## 📖 Overview
This project implements a novel **Multi-Modal Scene Understanding Pipeline** that integrates Object Detection (DETR), Vision-Language Captioning (BLIP-2), and Large Language Model Reasoning (Qwen 2.5 / Llama-3).

Unlike standard image captioning which often misses details, this pipeline:
1.  **Detects** individual objects in a scene.
2.  **Extracts** high-resolution patches for each object.
3.  **Generates** local descriptions for every patch.
4.  **Synthesizes** a global, reasoning-based scene description using an LLM.

## 🚀 Quick Start
This project is contained in a single, self-executing Jupyter Notebook for maximum reproducibility.

1.  **Open the Notebook:**
    [`Multi_Modal_Scene_Understanding.ipynb`](./Multi_Modal_Scene_Understanding.ipynb)
    
2.  **Environment Setup:**
    The notebook is optimized for **Google Colab (A100 GPU)**.
    Dependencies are listed in `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

## 📊 Methodology
The pipeline moves through four distinct phases:
* **Phase 1 (Detection):** Utilizing `DETR-ResNet-50` to identify objects.
* **Phase 2 (Patching):** Dynamic cropping and upscaling of object regions.
* **Phase 3 (Local Captioning):** Using `BLIP-2 (Opt-2.7b)` for detailed object-level visual descriptions.
* **Phase 4 (Reasoning):** Aggregating insights using `Qwen-2.5-7B-Instruct` (4-bit quantized) to resolve visual conflicts.

## 📈 Results
The notebook includes a comprehensive evaluation suite comparing our pipeline against baseline end-to-end models using:
* **CLIP Score:** Measuring visual alignment accuracy.
* **BERTScore:** Measuring semantic similarity to human ground truth.
* **Noun Density:** Quantifying information content.

## 👥 Authors
* [Your Name]
* [Teammate Name]
