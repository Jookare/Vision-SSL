# Vision-SSL
Contains implementations of popular self-supervised learning methods used in computer vision

## Navigation
The methods have been divided into `Generative`, `Invariance`, and `Other` categories. 
- `Generative`: methods mask a portion of the image and then try to learn to reconstruct the missing part of the image. 
- `Invariance`: methods apply different augmentations to generate multiple views from same image and then try to match the embeddings. 
- `Other`: methods contains methods that do not explicitly fit either.

Currently implemented methods include the following models:


| Model          | Year |  Category |
|----------------|------|-----------|
|MoCo            | 2019 | Invariance | 
|SimCLR          | 2020 | Invariance | 