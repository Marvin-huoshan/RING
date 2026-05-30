# RING

**Anonymous repository for the submission to IEEE S&P 2027:** *Your Privacy My Cloak: Backdoor Attack on Differentially Private Federated Learning*

## Description

**RING** is a collusion-based backdoor attack for **Differentially Private Federated Learning (DP-FL)**. It enables malicious clients to collaboratively inject backdoors while maintaining high attack effectiveness under DP noise, even in the presence of state-of-the-art defense mechanisms.

By evaluating multiple defenses and comparing against baseline attacks, this repository demonstrates aggravated privacy and security risks in DP-FL systems and underscores the urgent need for stronger, DP-aware mitigation strategies.

## Contents

This repository contains one main python file and one environment configuration file:

- `BD_Attack_Collusion.py`: Main implementation for evaluating the RING attack under DP-FL against state-of-the-art defenses.
- `environment.yml`: Conda environment specification listing all required dependencies.

## Running example

### Environment Installation

1. **Install Miniconda** (a tool for managing Python versions and virtual environments):

   - Download the installer for your system from [Anaconda's download page](https://www.anaconda.com/download/success)  
     or use:
     ```bash
     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
     ```
   - Run the installer (replace `Miniconda3-latest-Linux-x86_64.sh` with the name of your downloaded file):
     ```bash
     sh Miniconda3-latest-Linux-x86_64.sh
     ```
   - Verify the installation:
     ```bash
     conda -V
     ```

2. **Create the Conda Environment:**

   Navigate to the directory containing `environment.yml` and run:
   ```bash
   conda env create -f environment.yml

3. **Activate the environment:**

    ```bash
     conda activate FL_DP
     ```
   
### Running the Experiments

#### RING Attack Example

```bash
python BD_Attack_Collusion.py \
  --dataset mnist \
  --lr 0.05 \
  --dp_mechanism MA \
  --dp_epsilon 5 \
  --dp_delta 1e-5 \
  --dp_clip 10 \
  --dp_sample 1 \
  --attack_type Collusion \
  --frac 0.25 \
  --num_attacker 6 \
  --attack \
  --epochs 100 \
  --bs 500 \
  --local_ep 5 \
  --defense Deepsight \
  --gpu 1 \
  --PDR 0.5 \
  --iid qty \
  --re_weight
```

Additional backdoor baselines can be selected with --backdoor_baseline:

bash --backdoor_baseline standard --backdoor_baseline DBA --backdoor_baseline Neurotoxin 

The default value is standard, which preserves the original visible-trigger / RING behavior. Results are saved under Results, Results_DBA, and Results_Neuro for standard, DBA, and Neurotoxin, respectively.

###  Arguments

#### Dataset and Model

- `--dataset`  
  Specifies the dataset:
  - `mnist` — MNIST  
  - `cifar` — CIFAR-10  
  - `cifar100` — CIFAR-100 (`prob` split unsupported)
  - `sent140` — Sentiment-140  

- `--model`  
  Training model architecture:
  - `cnn` — for image datasets (default)  
  - `lstm` — for Sentiment-140  

---

#### Training Parameters

- `--lr`  
  Learning rate.

- `--epochs`  
  Number of federated training rounds.

- `--bs`  
  Local batch size.

- `--local_ep`  
  Number of local training epochs per client.

- `--frac`  
  Fraction of clients participating in each training round.

---

#### Differential Privacy Parameters

- `--dp_mechanism`  
  Differential privacy mechanism (`Laplace`, `Gaussian`, or `MA`).

- `--dp_epsilon`  
  Privacy budget ε.

- `--dp_delta`  
  Privacy parameter δ.

- `--dp_clip`  
  Gradient clipping bound \( C \) in DP-FL.

- `--dp_sample`  
  Client sampling ratio for DP.

---

#### Attack Configuration

- `--attack`  
  Enable the backdoor attack.

- `--attack_type`  
  Type of attack:
  - `Input` — DP-opt-in  
  - `Output` — DP-opt-out  
  - `Collusion` — **RING (proposed attack)**  

- `--backdoor_baseline`
  Backdoor baseline (`standard`, `DBA`, or `Neurotoxin`). Outputs are saved under `Results`, `Results_DBA`, and `Results_Neuro`, respectively. DBA and Neurotoxin currently support MNIST + CNN only; CIFAR100 supports the standard visible-trigger / Collusion path.

- `--num_attacker`  
  Number of malicious clients per round.

- `--PDR`  
  Poisoning Data Ratio.

---

#### Defense Methods

- `--defense`  
  Defense mechanism to deploy:
  - `Deepsight`  
  - `Krum`  
  - `Flame`  
  - `FLShield`  
  - `FreqFed`  
  - `MESAS`  

---

#### System and Utility Options

- `--gpu`  
  GPU index used for training.

- `--iid`  
  Non-IID data distribution type (e.g., `qty`).

- `--re_weight`  
  Reweight client updates according to dataset size.

- `--serial`  
  Enable serial execution to reduce GPU memory usage when using **Opacus**.
