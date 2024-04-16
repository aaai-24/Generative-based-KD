# Generative Model-based Feature Knowledge Distillation for Action Recognition 

This is the official repo of article [*Generative Model-based Feature Knowledge Distillation for Action Recognition*](https://ojs.aaai.org/index.php/AAAI/article/view/29473) which is accepted by AAAI-24.

## Abstract

Knowledge distillation (KD), a technique widely employed in computer vision, has emerged as a de facto standard for improving the performance of small neural networks. However, prevailing KD-based approaches in video tasks primarily focus on designing loss functions and fusing cross-modal information. This overlooks the spatial-temporal feature semantics, resulting in limited advancements in model compression. ddressing this gap, our paper introduces an innovative knowledge distillation framework, with the generative model for training a lightweight student model. In particular, the framework is organized into two steps: the initial phase is Feature Representation, wherein a generative modelbased attention module is trained to represent feature semantics; Subsequently, the Generative-based Feature Distillation phase encompasses both Generative Distillation and Attention Distillation, with the objective of transferring attentionbased feature semantics with the generative model. The efficacy of our approach is demonstrated through comprehensive experiments on diverse popular datasets, proving considerable enhancements in video action recognition task. Moreover, the effectiveness of our proposed framework is validated in the context of more intricate video action detection task.

## summary

* Design a novel attention module that leverages the generative model to represent feature semantics within the 3D-CNN architecture.
* Build a new framework that firstly introduces the novel concept of utilizing a generative model for distilling attention-based feature knowledge. 



## Getting Started

### Environment

- Python 3.7
- PyTorch == 1.4.0 **(Please make sure your pytorch version is 1.4)**
- NVIDIA GPU

### Setup

```shell
cd detection
pip3 install -r requirements.txt
python3 setup.py develop
```

### Data Preparation

#### For recognition

We follow the data preparation of repos [PyTorch implementation of popular two-stream frameworks for video action recognition](https://github.com/bryanyzhu/two-stream-pytorch) and [Video Dataset Preprocess](https://github.com/Emily0219/video-dataset-preprocess), please star them if helpful.

Environment we need to process dataset:

* OS: Ubuntu 16.04
* Python: 3.5
* CUDA: 8.0
* OpenCV3
* dense_flow

To successfully install [dense_flow](https://github.com/yjxiong/dense_flow/tree/opencv-3.1)(branch opencv-3.1), you probably need to install opencv3 with [opencv_contrib](https://github.com/opencv/opencv_contrib). (For opencv-2.4.13, dense_flow will be installed more easily without opencv_contrib, but you should run code of this repository under opencv3 to avoid error)

##### UCF101

Download data [UCF101](http://crcv.ucf.edu/data/UCF101.php) and use `unrar x UCF101.rar` to extract the videos.

Convert video to frames and extract optical flow

```
python build_of.py --src_dir ./UCF-101 --out_dir ./ucf101_frames --df_path <path to dense_flow>
```

build file lists for training and validation

```
python build_file_list.py --frame_path ./ucf101_frames --out_list_path ./settings
```

##### HMDB51

- Download videos and train/test splits [here](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/). Make sure to put the video files as the following structure:

```
  HMDB51
  ├── brush_hair
  │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0.avi
  │   └── ...
  ├── cartwheel
  │   ├── (Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi
  │   └── ...
  ├── catch
  │   ├── 96-_Torwarttraining_1_catch_f_cm_np1_le_bad_0.avi
  │   └── ...
```

- Convert from avi to jpg files using `utils/video_jpg_ucf101_hmdb51.py`

```
python utils/video2jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
```

- Generate n_frames files using `utils/n_frames_ucf101_hmdb51.py`

```
python utils/n_frames_ucf101_hmdb51.py jpg_video_directory
```

- Generate annotation file in txt format using

```
utils/hmdb_gen_txt.py
```

- `annotation_dir_path` includes brush_hair_test_split1.txt, ...

```
python utils/hmdb_gen_txt.py annotation_dir_path jpg_video_directory outdir
```

After pre-processing, the image output dir's structure is as follows:

```shell
  hmdb51_n_frames
  ├── brush_hair
  │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0
  │   │   ├── image_00001.jpg
  │   │   ├── ...
  │   │   └── n_frames
  │   └── ...
  ├── cartwheel
  │   │   ├── image_00001.jpg
  │   │   ├── ...
  │   │   └── n_frames
  │   └── ...
  ├── catch
  │   ├── 96-_Torwarttraining_1_catch_f_cm_np1_le_bad_0
  │   │   ├── image_00001.jpg
  │   │   ├── ...
  │   │   └── n_frames
  │   └── ...
```

The Train_Test split file contains following structure:

```shell
  hmdb51_TrainTestlist
  ├── hmdb51_train.txt
  ├── hmdb51_test.txt
  └── hmdb51_val.txt
```

#### For detection

we follow the data preparation of [AFSD](https://github.com/TencentYoutuResearch/ActionDetection-AFSD), please star it if helpful.

- **THUMOS14 RGB data:**

1. Download pre-processed RGB npy data (13.7GB): [Weiyun](https://share.weiyun.com/bP62lmHj)
2. Unzip the RGB npy data to `./datasets/thumos14/validation_npy/` and `./datasets/thumos14/test_npy/`

- **THUMOS14 flow data:**

1. Because it costs more time to generate flow data for THUMOS14, to make easy to run flow model, we provide the pre-processed flow data in Google Drive and Weiyun (3.4GB):[Google Drive](https://drive.google.com/file/d/1e-6JX-7nbqKizQLHsi7N_gqtxJ0_FLXV/view?usp=sharing),[Weiyun](https://share.weiyun.com/uHtRwrMb)  
2. Unzip the flow npy data to `./datasets/thumos14/validation_flow_npy/` and `./datasets/thumos14/test_flow_npy/`


**If you want to generate npy data by yourself, please refer to the following guidelines:**

- **RGB data generation manually:**

1. To construct THUMOS14 RGB npy inputs, please download the THUMOS14 training and testing videos. 
   Training videos: https://storage.googleapis.com/thumos14_files/TH14_validation_set_mp4.zip 
   Testing videos: https://storage.googleapis.com/thumos14_files/TH14_Test_set_mp4.zip 
   (unzip password is `THUMOS14_REGISTERED`)  
2. Move the training videos to `./datasets/thumos14/validation/` and the testing videos to `./datasets/thumos14/test/`
3. Run the data processing script: `python3 AFSD/common/video2npy.py configs/thumos14.yaml`

- **Flow data generation manually:**

1. If you should generate flow data manually, firstly install the [denseflow](https://github.com/open-mmlab/denseflow).
2. Prepare the pre-processed RGB data.
3. Check and run the script: `python3 AFSD/common/gen_denseflow_npy.py configs/thumos14_flow.yaml`

### Inference

We provide pretrained models contain Top-I3D for UCF101&HMDB51 and Top-AFSD for THUMOS14 dataset:[Google Drive](https://drive.google.com/drive/folders/1qTo4EEMTGFm_iJOerLvRxkIwyZ7TbBd4?usp=sharing)

For UCF101:

```shell
cd recognition
python3 test_Top_I3D_att_fusion.py configs/stu_thumos14.yaml
```

for HMDB51:

```shell
cd recognition
python3 test_Top_I3D_att_fusion_hmdb51.py configs/stu_thumos14.yaml
```

For THUMOS14:

```shell
cd detection
# run RGB model
python3 GKD/thumos14/test.py configs/thumos14.yaml --checkpoint_path=models/thumos14/thumos14-rgb.ckpt --output_json=thumos14_rgb.json

# run flow model
python3 GKD/thumos14/test.py configs/thumos14_flow.yaml --checkpoint_path=models/thumos14/thumos14-flow.ckpt --output_json=thumos14_flow.json

# run fusion (RGB + flow) model
python3 GKD/thumos14/test.py configs/thumos14.yaml --fusion --output_json=thumos14_fusion.json

# evaluate THUMOS14 fusion result as example
python3 GKD/thumos14/eval.py output/thumos14_fusion.json
```

### Training

For UCF101:

```shell
cd recognition
python3 train_Top_I3D_KD.py configs/stu_thumos14.yaml
```

For HMDB51:

```
cd recognition
python3 train_Top_I3D_KD_hmdb51.py configs/stu_thumos14.yaml
```

For THUMOS14:

```shell
cd detection
# train the RGB model
python3 GKD/thumos14/train.py configs/thumos14.yaml --lw=10 --cw=1 --piou=0.5

# train the flow model
python3 GKD/thumos14/train.py configs/thumos14_flow.yaml --lw=10 --cw=1 --piou=0.5
```

