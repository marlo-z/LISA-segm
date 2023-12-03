# LISA-segm

## Installation
```
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Dataset Structure
In the root directory of this repo, create the following directory structure
```
├── dataset
│   ├── refer_seg
│   │   ├── images
│   │   |   └── mscoco
│   │   |       └── images
│   │   |           └── train2017
│   │   ├── refcoco
│   │   |   └── instances.json
│   │   |   └── 'refs(google).p'
│   │   |   └── 'refs(unc).p'
│   │   ├── bbox
│   │   |   └── train
│   │   |   └── val

```
`bbox/train`: this directory should contain all the `.json` files from `/shared/kkkhanl/train` located on the Berkeley cluster.

## Training
`./kk_train_script.sh`
