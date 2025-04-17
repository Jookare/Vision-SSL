# Vision-SSL
This repository contains clean, modular implementations of popular **self-supervised learning (SSL)** methods for computer vision.

---

## Navigation
Methods are organized into the following categories:

- **Generative**  
  Models that reconstruct masked parts of the input image (e.g., MAE, iGPT).

- **Invariance-based**  
  Models that enforce consistent embeddings across augmented views of the same image (e.g., SimCLR, MoCo).

- **Other**  
  Methods that do not explicitly fit into the above categories but are part of the broader SSL landscape (e.g., I-JEPA).


> [!Note]  
> All models are structured such that the `forward()` method returns the **encoder output** used in downstream tasks.  
> Pretraining components (e.g., projection heads, momentum encoders) are implemented separately in training logic.

---

## Implemented Methods
Currently implemented methods include the following models:

| Model   | Year | Category     |
|---------|------|--------------|
| MoCo    | 2019 | Invariance   |
| SimCLR  | 2020 | Invariance   |

More methods coming soon...

## 📁 Project Structure
```
. 
├── generative/ 
│   └── <Method>/ 
├── invariance/ 
│   └── MoCo/ 
│   └── SimCLR/ 
└── images/ 
    └── <method>.png
```

Each method includes its own:
- `train.py` — for pretraining  
- `model.py` — with clean `forward()` for downstream use  
- `misc.py` — with other required functions