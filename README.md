Official PyTorch implementation of **Enhancing Cross-Domain Facial Expression Recognition with Expression Relationship Contrastive Learning in a Source-Free Setting**


## Project structure

The project structure is presented as follows

```
EERCL_in_Source-Free
├─ EERCL
│  ├─ __init__.py
│  ├─ hooks
│  ├─ models
│  ├─ loaders
│  └─ trainers
├─ basicda
├─ configs
│  ├─ _base_
│  │  ├─ cls_datasets
│  │  └─ cls_models
│  ├─ sfda_fer_class_relation
│  └─ vit-b
├─ data
├─ experiments
│  ├─ generate_random_port.py
│  ├─ get_visible_card_num.py
│  ├─ sfda_class_relation_train.sh
│  └─ sfda_source_only_train.sh
└─ train.py

```

**basicda**: basic framework for domain adaptation tasks

**configs**: training configs files for different experiments

**data**: contain dataset images and labels

**EERCL**: source code of our method, contains hooks (evaluation hooks), loaders (train and test loaders), models (
definition of models), trainers (training and testing process)

**experiments**: training scripts

**train.py**: entrance to training process

Below are the structure under **data**.

```
│fer/
├──ck
│  ├── angry
│  │   ├── ***01.jpg
│  │   ├── ***02.jpg
│  │   ├──......
│  ├── disgust
│  │   ├── ***01.jpg
│  │   ├── ***02.jpg
│  │   ├──......
│  ├── fear
│  │   ├── ......
│  ├── ......
├──expw
│  ├── angry
│  │   ├── ***01.jpg
│  │   ├── ***02.jpg
│  │   ├──......
│  ├── disgust
│  │   ├── ***01.jpg
│  ├── ......
│txt/
├──fer
│  ├── labeled_source_images_raf.txt
│  ├── labeled_target_images_raf.txt
│  ├── unlabeled_source_images_raf.txt
│  ├── unlabeled_target_images_raf.txt

```


## Getting Started

### Prerequisites
- Python 3.7
- PyTorch 1.10+
- 2+ NVIDIA GPUs (11GB+ VRAM recommended)

### 1. Data Preparation
Organize datasets following this structure:


### 2. Source Domain Training
```bash
# Single GPU training
CUDA_VISIBLE_DEVICES=0 bash experiments/sfda_source_only_train.sh \
    EXP_NAME  configs/sfda_fer_class_relation/sfda_fer_source_raf.py
```

After trained done, you can find the source pre-trained model in the runs folder. You should keep the path of model in mind for the target domain training. 

### 3. Configure Model Path
Edit`configs/sfda_fer_class_relation/class_relation_fer_AaD_rc.py`:


```
control = dict(
  save_interval=500,
  max_save_num=1,
  seed=2023,
  pretrained_model = SOURCE_MODEL.pth,
)
```
### 4. Target Domain Training
```bash
CUDA_VISIBLE_DEVICES=0,1 bash experiments/sfda_class_relation_train.sh \
    EXP_NAME configs/sfda_fer_class_relation/class_relation_fer_AaD_rc.py 
```
### 5. Monitor Results
View training logs
```bash
tail -f runs/EXP_NAME/logs/train.log
```

