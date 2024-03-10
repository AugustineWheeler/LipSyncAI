# LipSync.AI

Lipsync.ai is an implementation of the LIPNET model, a state-of-the-art deep learning approach for sentence-level lipreading. The original work was conducted by Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, and Nando de Freitas from the University of Oxford and Google DeepMind.

## Introduction

This repository contains the code implementation for Lipsync.ai, a powerful tool for end-to-end sentence-level lipreading. The project is based on the LIPNET model, which was developed by researchers at the University of Oxford and Google DeepMind.

## Features

- End-to-End Sentence-Level Lipreading
- Implementation of the LIPNET model
- User-Friendly End-to-End Lipreading Experience

### Installation

1. Clone the LipNet repository:

    ```
    git clone https://github.com/Arnav131003/LipSync.ai
    cd LipSync.ai
    pip install requirements.txt
    ```

2. Install dependencies:

      - Keras 2.0+
      - Tensorflow 2.0+
      - PIP (for package installation)
     - Streamlit


---

3 . Pretrained Model 

```
cd  model/checkpoint
```

4 . Using the pretrained model 
```
model.load("model/checkpoint")
```

## Inferencing 

<img width="1468" alt="Screenshot 2024-02-02 at 7 24 34 AM" src="https://github.com/Arnav131003/LipSync.ai/assets/75151775/160f8942-7b8d-480c-8391-d4e2fb6524d8">


    ```
    streamlit frontend.py 
    ```


## References

 https://arxiv.org/pdf/1611.01599.pdf?uuid=Fqbse38nqebdFpys3035

---


## Contributors 

- Augustine Wheeler

---
