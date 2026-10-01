"""Dataset and output folders in Colab, on Kaggle or on my own computer.

    from dataset_path import DATASET_FOLDER, OUTPUT_FOLDER
"""
import os
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parent

# Colab: Google Drive folder "My Drive/Dataset"
# https://drive.google.com/drive/folders/1npnGKO7qwgKPvfoKTclzeA59wfpm860g
# Outputs also go to Drive, so they are kept after the Colab runtime ends.
COLAB_FOLDER = Path('/content/drive/MyDrive/Dataset')
COLAB_OUTPUT_FOLDER = Path('/content/drive/MyDrive/KF_Net_outputs')

# Kaggle: dataset attached to the notebook ("Add Input"); all of /kaggle/input is searched if this is missing.
# Outputs go to /kaggle/working, the only writable folder that is kept with the notebook version.
KAGGLE_FOLDER = Path('/kaggle/input/datasets/elasphin/mknet-project')
KAGGLE_OUTPUT_FOLDER = Path('/kaggle/working/outputs')

# My computer: the Dataset and outputs folders next to this file
LOCAL_FOLDER = PROJECT_FOLDER / 'Dataset'
LOCAL_OUTPUT_FOLDER = PROJECT_FOLDER / 'outputs'


def get_folders() -> tuple[Path, Path]:
    """(dataset folder, output folder) of the current environment."""
    if Path('/kaggle').is_dir():                  # Kaggle (checked first: Kaggle also has google.colab installed)
        return (KAGGLE_FOLDER if KAGGLE_FOLDER.is_dir() else Path('/kaggle/input')), KAGGLE_OUTPUT_FOLDER
    if 'COLAB_RELEASE_TAG' in os.environ:          # Colab
        if not COLAB_FOLDER.parent.is_dir():       # Drive not mounted yet
            from google.colab import drive
            drive.mount('/content/drive')
        return COLAB_FOLDER, COLAB_OUTPUT_FOLDER
    return LOCAL_FOLDER, LOCAL_OUTPUT_FOLDER


DATASET_FOLDER, OUTPUT_FOLDER = get_folders()

if __name__ == '__main__':
    print('Dataset folder:', DATASET_FOLDER)
    for path in sorted(DATASET_FOLDER.iterdir()):
        print('  ', path.name)
    print('Output folder: ', OUTPUT_FOLDER)
