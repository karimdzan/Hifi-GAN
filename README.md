# DLA. Hifi-GAN implementation.

## Preparations
Make sure you run on Python version 3.8-3.10

1. Clone the repo.
    ```
    git clone https://github.com/karimdzan/Hifi-GAN.git
    ```

1. Download all necessery artifacts. This script will download LJSpeech dataset and some required packages.
    ```
    cd Hifi-GAN
    bash prepare.sh
    pip install -r requirements.txt
    ```

1. Download model checkpoint.
    ```
    gdown https://drive.google.com/file/d/1-gZrqzEzJZ4oWRPSMzpnpO2MXUmzGV5Y/view?usp=sharing
    ```

## Inference

1. You can use [inference.py](./inference.py) to synthesize a speech from mels:
    ```
    python3 inference.py --checkpoint path/to/checkpoint_7000.pth --data path/to/data --type="audio"
    ```
    The script above will create ./results folder that would store your synthesized audio.

## Training

1. To train your own model, use [train.py](./train.py):
    ```
    python3 train.py
    ```
   You will need to specify your wandb login key.

## Repository Structure

```bash
src/
├─model/                            # Folder with models
    ├─discriminator.py                  # Discriminators module
    ├─generator.py                      # Generator module
    ├─__init__.py                      
├─trainer/                          # Folder with training modules
    ├─inference.py                    # inference module
    ├─trainer.py                      # training module
    ├─utils.py                        # utils module
├─ config.py                        # Model configurations
├─ data.py                          # Wav and Mel dataset
├─ loss.py                          # Loss functions
├─ mel.py                           # Mel module
├─ wandb_writer.py                  # Wandb writer and logger
inference.py                     # Inference and speech generation script
train.py                         # Main training script
```