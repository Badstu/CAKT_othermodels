## Comparative Experiment

---

### Introduction

We use 8 models to compare with CAKT, namely DKT, DKVMN, SKVMN, SAKT, EKT, CKT, DKT-F (DKT+forgetting) and AKT.

Following the instructions, and you can get the results of these models quickly.

### Environment

These projects are developed using 

- python	3.7

- Pytorch    1.4.0

- Tensorflow    1.13.1

- visdom    0.1.8.9

- torchnet    0.0.4

- pandas    1.1.4

- tqdm    4.51.0

- numpy    1.19.2

- Pillow    8.0.1

- pytz    2020.4

- pyzmq    20.0.0

- CUDA    10.2

  on NVIDIA Titan RTX GPU. You'd better configure  the environment as this.

### Quick  start

---

#### 1. Clone the repo

```
git clone git@github.com:Badstu/CAKT.git
```

#### 2. Install dependencies

```
pip install -r requirements.txt
```

#### 3. Dataset

You can find the datasets at `dataset` folder, there are five datasets used in these projects.

The datasets are namely `'assist2009_updated'`, `'assist2015'`, `'assist2017'`, `'STATICS'`, `'synthetic'`. You can change the dataset name parameter to run on different dataset.

#### 4. Quick run

- ##### Deep Knowledge Tracing

  You can run DKT model with `main.py`.

  ```
  cd DKT
  python main.py --dataset dataset_name    # change the dataset_name as you need
  ```

- ##### Dynamic Key-Value Memory Networks for Knowledge Tracing

  You can run DKVMN model with `main.py`.

  ```
  cd DKVMN
  python main.py --dataset dataset_name    # change the dataset_name as you need
  ```

- ##### Sequential Key-Value Memory Networks

  You can run SKVMN model with `main.py`.

  ```
  cd SKVMN
  python main.py --dataset dataset_name    # change the dataset_name as you need
  ```

- ##### A Self-Attentive Model for Knowledge Tracing

  You can run SAKT model with `main.py`.

  ```
  cd SAKT
  python main.py --dataset dataset_name    # change the dataset_name as you need
  ```

- ##### Exercise-aware Knowledge Tracing

  You can run EKT model with `EKT_experiment.py`.

  ```
  cd EKT
  python EKT_experiment.py dataset_name    # change the dataset_name as you need
  ```

- ##### Convolutional Knowledge Tracing

  You can run CKT model with `train.py`.

  ```
  cd CKT
  python train.py dataset_name    # change the dataset_name as you need
  ```

  After the training process finished, You will see an instruction like this: 

  ![image-20201206114558459](C:\Users\zmx\AppData\Roaming\Typora\typora-user-images\image-20201206114558459.png)

  The number `1607160632` is the trained model ID.

  To test the model, you can run with `test.py`.

  ```
  python test.py model_id dataset_name    # the model_id corresponds to the ID in the instruction after the training process
  										# the dataset_name here should accord with the training model
  ```

  

- ##### Augmenting Knowledge Tracing by Considering Forgetting Behavior

  You can run DKT-F model with `main.py`.

  ```
  cd DKT_F
  python main.py --dataset dataset_name    # change the dataset_name as you need
  ```

- ##### Context-Aware Attentive Knowledge Tracing

  You can run AKT model with `main.py`.

  ```
  cd AKT
  python main.py --dataset dataset_name    # change the dataset_name as you need
  ```


