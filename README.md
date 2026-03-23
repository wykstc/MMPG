# MMPG
This is the official implementation for the paper ["MMPG: MoE-based Adaptive Multi-Perspective Graph Fusion for Protein Representation Learning"](https://ojs.aaai.org/index.php/AAAI/article/view/37096).


## Dependencies

- Python 3.10.14
- bio==1.7.1
- torch-2.1.0+cu121
-Torch-geometric 2.6.0

## Dataset
We use the four datasets, which can be found at:

https://github.com/DeepGraphLearning/torchdrug

https://github.com/divelab/DIG

To preprocess the data, please use the scripts from dataset fold.


## Usage

### Train the model

```
python train.py 
```

### Evaluate the model

The evaluation can be automatically implemented after training. You can also change the train.py to implement it individually.


## Citation

If you find our work useful, please consider citing it as follows:
```
@article{wang2026mmpg,
title={MMPG: MoE-based Adaptive Multi-Perspective Graph Fusion for Protein Representation Learning},
volume={40},
url={https://ojs.aaai.org/index.php/AAAI/article/view/37096},
DOI={10.1609/aaai.v40i2.37096},
journal={Proceedings of the AAAI Conference on Artificial Intelligence},
author={Wang, Yusong and Shen, Jialun and Wu, Zhihao and Xu, Yicheng and Tan, Shiyin and Xu, Mingkun and Wang, Changshuo and Song, Zixing and Tiwari, Prayag},
year={2026},
month={Mar.},
pages={1240-1248}
}
```

