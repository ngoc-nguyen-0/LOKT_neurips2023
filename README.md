# [Neurips-2023] Implementation of paper "Label-Only Model Inversion Attacks via Knowledge Transfer"

[Paper] | [Project page](https://ngoc-nguyen-0.github.io/lokt/)

## 1. Setup Environment
This code has been tested with Python 3.7, PyTorch 1.11.0 and Cuda 11.3. 

```
conda create -n MI python=3.7

conda activate MI

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt
```

## 2. Prepare Dataset & Checkpoints

* Dowload CelebA and FFHQ dataset at the official website.
- CelebA: download and extract the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Then, place the `img_align_celeba` folder to `.\datasets\celeba`

- FFHQ: download and extract the [FFHQ](https://github.com/NVlabs/ffhq-dataset). Then, place the `thumbnails128x128` folder to `.\datasets\ffhq`

* Modify the arguments ```img_priv_path``` and ```img_pub_path`` of the config at ```./config/dataset/dataset_name.json```

* We use the same target models as previous papers. You can download target models at https://drive.google.com/drive/folders/1kq4ArFiPmCWYKY7iiV0WxxUSXtP70bFQ?usp=sharing


## 3. Train TACGAN
* Modify the arguments of the config at ```./config/exp/experiment_name.json```. Important arguments:
  * `result_dir`: Modify the output path.
  * `path_T`: Modify the path to the target model
    
**Other arguments will be automatically updated when you train the TACGAN and surrogate models.**

* Then, run the following command line to get the TACGAN:

```
python train_tacgan.py \
--alpha=1.5 \
--cGAN \
--config_exp ./config/exp/FaceNet64_celeba.json \
--is_wandb 
```

## 4. Train surrogate model
* Generate images by TACGAN to train the surrogate models:

```
python create_dataset.py \
--config_exp ./config/exp/FaceNet64_celeba.json 
```

* Then, run the following command line to get the surrogate model:
  
```
python train_surrogate_model.py \
--is_wandb \
--config_exp ./config/exp/FaceNet64_celeba.json \
--surrogate_model_id 0 
```

Modify ```surrogate_model_id``` to change the architectures of the surrogate model. We provide 3 architectures for surrogate models:
* 0: Densenet121
* 1: Densenet161
* 2: Densenet169

* You can download the surrogate models and TACGAN at: 
https://drive.google.com/drive/folders/11gvFaltjILgPmNgkJ_w1fcINj1RlfOuw?usp=sharing
  
## 5. Attack and evaluation
* Important arguments:
  * `inv_loss_type`: select the identity loss ***margin*** or ***ce***
  * `classid` select the surrogate models ***0***, ***1***, ***2***, or ***0,1,2***

* Run the following command line to attack:

```
python plg_tacgan.py \
--inv_loss_type=margin \
--save_dir='results_facenet64' \
--classid='0,1,2' \
--config_exp ./config/exp/FaceNet64_celeba.json \
```



* Run the following command line to evaluate:

```
python evaluation.py \
--save_dir='results_facenet64' \
--classid='0,1,2' \
--config_exp ./config/exp/FaceNet64_celeba.json \

```


## Acknowledgements
We gratefully acknowledge the following works:
- LOMMA: https://ngoc-nguyen-0.github.io/re-thinking_model_inversion_attacks/
- PLGMI: https://github.com/LetheSec/PLG-MI-Attack/tree/main
- Densenet (Pytorch): https://pytorch.org/vision/main/models/densenet.html
- Experiment Tracking with Weights and Biases : https://www.wandb.com/


## Reference
<a id="1">[1]</a> 
Yuan, Xiaojian, et al. "Pseudo Label-Guided Model Inversion Attack via Conditional Generative Adversarial Network." AAAI (2023).






