# Referring Parser within Strefer

**[Salesforce AI Research](https://www.salesforceairesearch.com/)**

[Honglu Zhou](https://sites.google.com/view/hongluzhou/), [Xiangyu Peng](https://xiangyu-peng.github.io/), [Shrikant Kendre](https://www.linkedin.com/in/skendre), [Michael S. Ryoo](http://michaelryoo.com/), [Silvio Savarese](https://www.linkedin.com/in/silvio-savarese-97b76114/), [Caiming Xiong](http://cmxiong.com/), [Juan Carlos Niebles](https://www.niebles.net/)



## Installation ⚙️
Create a conda environment `vllm`:
```bash
conda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm==0.7.2
```

## Getting Started 🚀
### Entity to General Noun
Follow the guidelines below to try our code that converts complex entity-referring expressions into their generalized nouns:
```bash
conda activate vllm
cd data-engine/referring-parser
python entity_to_general_noun.py
```

