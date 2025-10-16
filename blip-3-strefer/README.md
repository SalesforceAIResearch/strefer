# BLIP-3-Strefer

**[Salesforce AI Research](https://www.salesforceairesearch.com/)**

[Honglu Zhou](https://sites.google.com/view/hongluzhou/), [Xiangyu Peng](https://xiangyu-peng.github.io/), [Shrikant Kendre](https://www.linkedin.com/in/skendre), [Michael S. Ryoo](http://michaelryoo.com/), [Silvio Savarese](https://www.linkedin.com/in/silvio-savarese-97b76114/), [Caiming Xiong](http://cmxiong.com/), [Juan Carlos Niebles](https://www.niebles.net/)





## Highlights 🌟

Our model incorporates a spatiotemporal object encoder from [Video-
Refer](https://arxiv.org/abs/2501.00599) for **region comprehension** and temporal tokens from [Grounded-VideoLLM](https://arxiv.org/abs/2410.03290) for **precise timestamp comprehension**. These modules are integrated into the [BLIP-3-Video](https://www.salesforceairesearch.com/opensource/xGen-MM-Vid/index.html) architecture. We call this model trained using our Strefer-synthesized instruction data as **BLIP-3-Strefer**.

Here, we release our code used to train BLIP-3-Strefer and perform inference.

<div align=center>
  <img src="https://raw.githubusercontent.com/SalesforceAIResearch/strefer/main/assets/strefer_model.png" width="800">
</div>


## Installation ⚙️
```bash
conda env create -f environment.yml
```
Please refer to [BLIP-3's installation guide](https://github.com/salesforce/LAVIS/tree/xgen-mm?tab=readme-ov-file#installation).

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
## Training 🧠
### Step 1: 
**Our model is initialized from BLIP-3 pretrained checkpoint. Download this pretrained checkpoint [here]().**

### Step 2: 
Update the paths and credentials (e.g., wandb api key) in `strefer/blip-3-strefer/open_flamingo/scripts/train.sh`.

### Step 3: 
```bash
cd strefer
bash blip-3-strefer/open_flamingo/scripts/train.sh
```

## Inference 🤖
### Step 1: 
Use eithr the checkpoint you obtained after replicating our model training, or **download the BLIP-3-Strefer checkpoint [here]().**

### Step 2: 
Update the paths in `strefer/blip-3-strefer/open_flamingo/scripts/test.sh`.

### Step 3: 
```bash
cd strefer
bash blip-3-strefer/open_flamingo/scripts/test.sh
```

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