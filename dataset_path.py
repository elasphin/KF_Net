"""Path of the SmartPNT-POS dataset in Colab, on Kaggle or on my own computer.

    from dataset_path import DATASET_FOLDER
"""
import os
from pathlib import Path

# Colab: Google Drive folder "My Drive/Dataset"
# https://drive.google.com/drive/folders/1npnGKO7qwgKPvfoKTclzeA59wfpm860g
COLAB_FOLDER = Path('/content/drive/MyDrive/Dataset')

# Kaggle: dataset attached to the notebook ("Add Input")
KAGGLE_FOLDER = Path('/kaggle/input/datasets/elasphin/mknet-project')

# My computer: the Dataset folder next to this file
LOCAL_FOLDER = Path(__file__).resolve().parent / 'Dataset'


def get_dataset_folder() -> Path:
    if Path('/kaggle').is_dir():                  # Kaggle (checked first: Kaggle also has google.colab installed)
        return KAGGLE_FOLDER
    if 'COLAB_RELEASE_TAG' in os.environ:          # Colab
        if not COLAB_FOLDER.parent.is_dir():       # Drive not mounted yet
            from google.colab import drive
            drive.mount('/content/drive')
        return COLAB_FOLDER
    return LOCAL_FOLDER


DATASET_FOLDER = get_dataset_folder()

if __name__ == '__main__':
    print('Dataset folder:', DATASET_FOLDER)
    for path in sorted(DATASET_FOLDER.iterdir()):
        print('  ', path.name)
