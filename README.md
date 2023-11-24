# DLA. FastSpeech2 implementation.

## Preparations
Make sure you run on Python version 3.8 or 3.9

1. Clone the repo.
    ```
    git clone https://github.com/karimdzan/TTS.git
    ```

1. Download all necessery artifacts. This script will download LJSpeech dataset, its mels and alignments that are necessery for training.
    ```
    bash prepare_data.sh
    ```

1. Download model checkpoint.
    ```
    gdown https://drive.google.com/file/d/1dy2AYvLOUpv0FulOWVDJopPhwf09tuLL/view?usp=sharing
    ```

1. In order to download pitch and energy and move them to data folder, run [prepare_pitch_energy.sh](./prepare_pitch_energy.sh):
   ```
   bash prepare_pitch_energy.sh
   ```
   You can also compute pitch and energy yourself:
   ```
   python3 compute_pitch_energy.py
   ```

## Inference

1. You can use [test.py](./test.py) to synthesize a speech from [test.txt](./test.txt):
    ```
    python3 test.py --model data/download/data/cool_model/checkpoint_90000.pth.tar --text test.txt
    ```
    You can specify pitch, energy and duration parameters for audio control using same named arguments(--pitch etc., use --help function)
    The script above will create ./results folder that would store your synthesized audio.

## Training

1. To train your own model, use [train.py](./train.py):
    ```
    python3 train.py
    ```
   You will need to specify your wandb login key.

## Repository Structure

The main folder [src](./src) contains model and training classes, as well as waveglow vocoder implementation and its preprocessing tools.