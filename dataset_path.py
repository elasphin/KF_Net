"""Path of the SmartPNT-POS dataset in Colab, on Kaggle or on my own computer.

    from dataset_path import DATASET_FOLDER
"""
from pathlib import Path

# Colab: Google Drive folder https://drive.google.com/drive/folders/1npnGKO7qwgKPvfoKTclzeA59wfpm860g
# (change the name if the folder is called differently in "My Drive")
COLAB_FOLDER = Path('/content/drive/MyDrive/SmartPNT-POS')

# Kaggle: notebook with https://www.kaggle.com/datasets/fengzhusgg/smartpnt-pos attached ("Add Input")
KAGGLE_FOLDER = Path('/kaggle/input/smartpnt-pos')

# My computer: the Dataset folder next to this file
LOCAL_FOLDER = Path(__file__).resolve().parent / 'Dataset'


def get_dataset_folder() -> Path:
    try:
        from google.colab import drive    # only exists in Colab
    except ImportError:
        pass
    else:
        drive.mount('/content/drive')
        return COLAB_FOLDER
    if Path('/kaggle/input').is_dir():
        return KAGGLE_FOLDER
    return LOCAL_FOLDER


DATASET_FOLDER = get_dataset_folder()

if __name__ == '__main__':
    print('Dataset folder:', DATASET_FOLDER)
    for path in sorted(DATASET_FOLDER.iterdir()):
        print('  ', path.name)
