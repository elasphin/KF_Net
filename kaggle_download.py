"""Find a SmartPNT-POS dataset folder, or download it from Kaggle.

Search order:
  1. Kaggle notebook with the dataset attached: files are read directly from
     /kaggle/input/smartpnt-pos (no download).
  2. A copy under ./Dataset (any depth).
  3. Download only the files this project needs with kagglehub. This needs a
     Kaggle API token (~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY).
"""
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

import settings as cfg

# Files of one dataset folder used by this project ({imu} = IMU type from README.xml).
NEEDED_FILE_PATTERNS = ('README.xml', 'ROVE*GroundTruth.txt', 'Rove*GroundTruth.txt', '{imu}_GroundTruth.txt',
                        '{imu}.imr', 'ROVE*.*[oO]', '*.[sS][pP]3', '*.[cC][lL][kK]', '[bB][rR][dD][mM]*')


def find_dataset_folder(folder_name: str) -> Path:
    for root in (cfg.KAGGLE_INPUT_FOLDER, cfg.LOCAL_DATA_FOLDER):
        if root.is_dir():
            matches = sorted(p for p in root.rglob(folder_name) if p.is_dir())
            if matches:
                return matches[0]
    return download_dataset_folder(folder_name)


def list_kaggle_files() -> list[str]:
    from kagglehub.clients import build_kaggle_client
    from kagglesdk.datasets.types.dataset_api_service import ApiListDatasetFilesRequest

    owner, dataset = cfg.KAGGLE_DATASET.split('/')
    names, page_token = [], None
    with build_kaggle_client() as client:
        while True:
            request = ApiListDatasetFilesRequest()
            request.owner_slug, request.dataset_slug, request.page_size = owner, dataset, 200
            if page_token:
                request.page_token = page_token
            response = client.datasets.dataset_api_client.list_dataset_files(request)
            names += [f.name for f in response.dataset_files]
            page_token = response.next_page_token
            if not page_token:
                return names


def download_dataset_folder(folder_name: str) -> Path:
    import kagglehub
    from read_dataset import read_rover_info

    files = [n for n in list_kaggle_files() if folder_name in PurePosixPath(n).parts]
    if not files:
        raise FileNotFoundError(f'{folder_name} was not found in the Kaggle dataset {cfg.KAGGLE_DATASET}. '
                                f'Set TRAIN_FOLDER_NAME / TEST_FOLDER_NAME in settings.py.')

    def download(names):
        return [Path(kagglehub.dataset_download(cfg.KAGGLE_DATASET, path=n)) for n in names]

    readme = download([n for n in files if PurePosixPath(n).name == 'README.xml'])
    if not readme:
        raise FileNotFoundError(f'README.xml is missing in {folder_name} on Kaggle')
    imu_type = read_rover_info(readme[0])[0]
    patterns = [p.format(imu=imu_type) for p in NEEDED_FILE_PATTERNS]
    download([n for n in files if any(fnmatch(PurePosixPath(n).name, p) for p in patterns)])
    return readme[0].parent
