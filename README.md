# Strefer: Empowering Video LLMs with Space-Time Referring and Reasoning via Synthetic Instruction Data

<div align=center>

[![Homepage](https://img.shields.io/badge/Homepage-visit-9DC3E6)](https://strefer.github.io/) 
[![arXiv preprint](https://img.shields.io/badge/arxiv-2501.00599-ECA8A7?logo=arxiv)](https://arxiv.org/abs/2509.03501) 
[![Dataset](https://img.shields.io/badge/Dataset-Hugging_Face-E59FB6)](https://huggingface.co/datasets/strefer/strefer) 
[![Model](https://img.shields.io/badge/Model-download-E6A151)]() 
[![video](https://img.shields.io/badge/Watch_Video-36600E?logo=youtube&logoColor=green)](https://youtu.be/87L2tyzxlvs)
</div>

**[Salesforce AI Research](https://www.salesforceairesearch.com/)**

[Honglu Zhou](https://sites.google.com/view/hongluzhou/), [Xiangyu Peng](https://xiangyu-peng.github.io/), [Shrikant Kendre](https://www.linkedin.com/in/skendre), [Michael S. Ryoo](http://michaelryoo.com/), [Silvio Savarese](https://www.linkedin.com/in/silvio-savarese-97b76114/), [Caiming Xiong](http://cmxiong.com/), [Juan Carlos Niebles](https://www.niebles.net/)


<div align="center">
  <img src="https://raw.githubusercontent.com/SalesforceAIResearch/strefer/main/assets/our_work.gif" width="800">
</div>
<br>



## Highlights 🌟

This is the repository for our paper **Strefer: Empowering Video LLMs with Space-Time Referring and Reasoning via Synthetic Instruction Data**. 

<div align=center>
  <img src="https://raw.githubusercontent.com/SalesforceAIResearch/strefer/main/assets/teaser-strefer.png" width="800">
</div>
<br>


**Strefer** is a data engine that synthesizes instruction-response pairs through a scalable, grounded approach that enhances fine-grained spatial and temporal perception and reasoning over videos for tuning Video LLMs.

By design, **Strefer** generates instruction-response pairs—requiring no legacy annotations—based on its pseudo-annotated video metadata. It automatically clips the video into segments and pseudo-annotates the video metadata, including active entities, their locations (as masklets), and action timelines, for complex video scenarios, such as scenes containing multiple entities of the same category, and cases where entities do not appear in the first frame, or temporarily exit and re-enter the frame.

<div align=center>
  <img src="https://raw.githubusercontent.com/SalesforceAIResearch/strefer/main/assets/method.png" width="800">
</div>
<br>


**Strefer** enhances the ability of Video LLMs to interpret spatial and temporal references, fostering more versatile, space-time-aware reasoning essential for real-world AI companions.


<div align="center">
  <img src="https://raw.githubusercontent.com/SalesforceAIResearch/strefer/main/assets/new_capability.gif" width="800">
</div>

<br>

## Download Strefer-Synthesized Instruction Data 💾
We release the **final recipe descirbed in our [paper](https://arxiv.org/abs/2509.03501)** including the Strefer synthesized instruction-response pairs on Hugging Face: https://huggingface.co/datasets/strefer/strefer

The following image illustrates the data composition of the final recipe used in our experiments:

<br>
<div align="center">
  <img src="https://raw.githubusercontent.com/SalesforceAIResearch/strefer/main/assets/data_final recipe_pie_chart.png" width="800">
</div>
<br>

If you hope to replicate model training, please follow the following steps to download and prepare the data:
```bash
git clone --recurse-submodules https://github.com/SalesforceAIResearch/strefer.git
cd blip-3-strefer
mkdir data
cd data
git clone https://huggingface.co/datasets/strefer/strefer
apt-get install git-lfs
git lfs install
git lfs pull
```

<br>

## Code for Model Training and Inference 🧠
Our model incorporates a spatiotemporal object encoder from [Video-
Refer](https://arxiv.org/abs/2501.00599) for **region comprehension** and temporal tokens from [Grounded-VideoLLM](https://arxiv.org/abs/2410.03290) for **precise timestamp comprehension**. These modules are integrated into the [BLIP-3-Video](https://www.salesforceairesearch.com/opensource/xGen-MM-Vid/index.html) architecture. We call this model trained using our Strefer-synthesized instruction data as **BLIP-3-Strefer**.

We release our code used to train BLIP-3-Strefer and perform inference. For detailed guidelines, please refer to [this README](blip-3-strefer/README.md).

<div align=center>
  <img src="https://raw.githubusercontent.com/SalesforceAIResearch/strefer/main/assets/strefer_model.png" width="800">
</div>
<br>

## Code for Referring Masklet Generation 💻

Our novel **Referring Masklet Generation Pipeline** is a key module within **Strefer**. This pipeline produces tracked segmentation masks from videos with complex structures based on multi-word natural language referring expressions. 


Our referring masklet generator is carefully crafted to address key limitations overlooked by prior works by orchestrating complementary strengths of the state-of-the-art pixel-level vision foundation models to achieve more effective results. The code has been released; for detailed installation and usage guidelines, please refer to [this README](data-engine/referring-masklet-generator).

<br>
<div align=center>
  <img src="https://raw.githubusercontent.com/SalesforceAIResearch/strefer/main/assets/teaser-referring_masklet_generator.png" width="800">
</div>
<br>

## License 💼
Our code, data, and models are released for research-only, non-commercial purposes under a CC-BY-NC 4.0 license. Users are responsible for making their own assessment of any obligations or responsibilities under the corresponding licenses or the terms and conditions applicable to the original code, data, and model weights.

<br>

## Citation 📝
Please cite us if you find our work helpful. Thank you! 🥰🙏💖
```bibtex
@article{zhou2025strefer,
  title={Strefer: Empowering Video LLMs with Space-Time Referring and Reasoning via Synthetic Instruction Data},
  author={Zhou, Honglu and Peng, Xiangyu and Kendre, Shrikant and Ryoo, Michael S. and Savarese, Silvio and Xong, Caiming and Niebles, Juan Carlos},
  journal={arXiv preprint arXiv:2509.03501},
  year={2025}
}
```
