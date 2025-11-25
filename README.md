# MMPG
This is the official implementation for the paper "M^2PG: MoE-based Adaptive Multi-Perspective Graph Fusion for Protein Representation Learning".


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





