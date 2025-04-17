# Momentum Contrast

This implementation contains the code for **Momentum Contrast (MoCo)** v3.
- MoCo v1: https://arxiv.org/abs/1911.05722 published in CVPR 2020.
- MoCo v2: https://arxiv.org/abs/2003.04297
- MoCo v3: https://arxiv.org/abs/2104.02057 published in ICCV 2021.

For a conceptual overview, check out for example this [Medium post](https://medium.com/data-science/from-moco-v1-to-v3-towards-building-a-dynamic-dictionary-for-self-supervised-learning-part-1-745dc3b4e861).

---

## Architecture

![Momentum Contrast v3 architecture diagram](../../images/MoCo.png)


## Usage

Train MoCo v3 using the `train.py` script:

Pretrain with:

```bash
python train.py 
```

After pretraining
- ✅ Keep the query encoder
- ❌ Discard the momentum encoder and the projection head
