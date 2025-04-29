# 🌀 MoCo

*Momentum Contrast for Unsupervised Visual Representation Learning*

This folder contains the implementation of **MoCo (Momentum Contrast)**, a contrastive learning framework with a momentum-updated encoder.

> 📄 Papers\
> v1: https://arxiv.org/abs/1911.05722 published in CVPR 2020.\
> v2: https://arxiv.org/abs/2003.04297\
> v3: https://arxiv.org/abs/2104.02057 published in ICCV 2021.


## Overview
MoCo learns representations by contrasting positive pairs (different views of the same image) against a large set of negative examples. It introduces a momentum encoder to maintain a consistent dictionary for contrast.

For a conceptual breakdown: 
- [Review: MoCo v1 - Medium](https://sh-tsang.medium.com/review-moco-momentum-contrast-for-unsupervised-visual-representation-learning-99b590c042a9).
- [Review: MoCo v2 - Medium](https://sh-tsang.medium.com/review-moco-v2-improved-baselines-with-momentum-contrastive-learning-8b03598a0324)
- [Review: MoCo v3 - Medium](https://sh-tsang.medium.com/review-moco-v3-an-empirical-study-of-training-self-supervised-vision-transformers-4a1a70c53ecd)

or

- [From MoCo v1 to v3 - Medium](https://medium.com/data-science/from-moco-v1-to-v3-towards-building-a-dynamic-dictionary-for-self-supervised-learning-part-1-745dc3b4e861).


---

## Architecture

![MoCo v3 architecture diagram](../../assets/MoCo.png)

- $x_1$, $x_2$: two augmented views of the same image
- Two encoders:
    - **Query encoder** (updated via backprop)
    - **Key encoder** (updated via momentum)
- Contrastive loss pulls together $(x_1, x_2)$ pairs
## Usage

Start pretraining with:
```bash
python train.py 
```
Once pretraining is finished:
- ✅ Keep the query encoder for downstream tasks
- ❌ Discard the momentum encoder and the projection head
