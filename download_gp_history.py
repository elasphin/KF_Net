"""Download historical TLEs (Space-Track ``gp_history``) for the LEO simulation.

Consecutive-TLE differencing needs, for every satellite, TLEs published before
each dataset window (estimator) and TLEs around the window centre (truth).
This script fetches all element sets of the given NORAD IDs in
[window start - PAD_BEFORE_DAYS, window end + PAD_AFTER_DAYS] for both
datasets and writes them to Dataset/LEO_TLE in 3LE format.  TLEProvider
removes duplicates, so re-downloading overlapping spans is harmless.

Credentials are read from the environment:
    SPACETRACK_USER, SPACETRACK_PASSWORD

Space-Track asks users to query gp_history sparingly: run this once, keep the
files, and do not schedule it.  Requests are batched and rate limited below
Space-Track's published limits (30 requests/minute, 300 requests/hour).

Usage:
    python download_gp_history.py                # IDs already in LEO_TLE
    python download_gp_history.py 43072 44057    # explicit NORAD IDs
"""
from __future__ import annotations

from datetime import timedelta
from http.cookiejar import CookieJar
from pathlib import Path
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import config as cfg
from navigation_models import GPS_EPOCH, GPS_UTC_LEAP_S
from simulation_data import load_truth, unique_file

BASE_URL = 'https://www.space-track.org'
PAD_BEFORE_DAYS = cfg.TLE_MAX_AGE_DAYS + 1.0
PAD_AFTER_DAYS = cfg.TLE_TRUTH_MAX_OFFSET_DAYS + 1.0
IDS_PER_REQUEST = 50
SECONDS_BETWEEN_REQUESTS = 3.0


def dataset_window_utc(root: Path):
    truth_path = unique_file(
        root,
        ('ROVE_GroundTruth.txt', 'ROVE_01_GroundTruth.txt', 'Rove_01_GroundTruth.txt'),
        'rover 01 ground truth',
    )
    time_gpst_s = load_truth(truth_path).time
    to_utc = lambda t: GPS_EPOCH + timedelta(seconds=float(t) - GPS_UTC_LEAP_S)
    return to_utc(time_gpst_s[0]), to_utc(time_gpst_s[-1])


def existing_norad_ids(directory: Path) -> list[int]:
    ids = set()
    for path in directory.glob('*'):
        if not path.is_file():
            continue
        for line in path.read_text(encoding='ascii', errors='replace').splitlines():
            if re.match(r'^1 \d{5}', line):
                ids.add(int(line[2:7]))
    excluded = {int(item.split('-')[1]) for item in cfg.LEO_EXCLUDED_SAT_IDS}
    return sorted(ids - excluded)


def open_session():
    user = os.environ.get('SPACETRACK_USER')
    password = os.environ.get('SPACETRACK_PASSWORD')
    if not user or not password:
        raise SystemExit('set SPACETRACK_USER and SPACETRACK_PASSWORD first')
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    data = urllib.parse.urlencode({'identity': user, 'password': password}).encode()
    with opener.open(f'{BASE_URL}/ajaxauth/login', data=data, timeout=60) as response:
        body = response.read().decode(errors='replace')
    if 'Failed' in body:
        raise SystemExit(f'Space-Track login failed: {body}')
    return opener


def query_gp_history(opener, norad_ids: list[int], start, end) -> str:
    path = '/'.join([
        'basicspacedata', 'query', 'class', 'gp_history',
        'NORAD_CAT_ID', ','.join(str(i) for i in norad_ids),
        'EPOCH', f"{start:%Y-%m-%dT%H:%M:%S}--{end:%Y-%m-%dT%H:%M:%S}",
        'orderby', urllib.parse.quote('NORAD_CAT_ID asc,EPOCH asc'),
        'format', '3le',
        'emptyresult', 'show',
    ])
    with opener.open(f'{BASE_URL}/{path}', timeout=300) as response:
        return response.read().decode('ascii', errors='replace')


def write_by_satellite(text: str, tag: str) -> int:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    blocks: dict[int, list[str]] = {}
    for i in range(len(lines) - 1):
        if lines[i].startswith('1 ') and lines[i + 1].startswith('2 '):
            name = lines[i - 1] if i > 0 and lines[i - 1].startswith('0 ') else ''
            blocks.setdefault(int(lines[i][2:7]), []).extend(
                ([name] if name else []) + [lines[i], lines[i + 1]]
            )
    for norad, block in blocks.items():
        out = cfg.TLE_DIR / f'sat{norad:09d}_gp_history_{tag}.txt'
        out.write_text('\n'.join(block) + '\n', encoding='ascii')
    return len(blocks)


def main(argv: list[str]) -> None:
    norad_ids = [int(a) for a in argv] if argv else existing_norad_ids(cfg.TLE_DIR)
    if not norad_ids:
        raise SystemExit('no NORAD IDs to download')
    roots = [cfg.TRAIN_DIR, cfg.TEST_DIR]
    opener = open_session()
    for root in roots:
        start, end = dataset_window_utc(root)
        start -= timedelta(days=PAD_BEFORE_DAYS)
        end += timedelta(days=PAD_AFTER_DAYS)
        tag = f'{start:%Y%m%d}_{end:%Y%m%d}'
        written = 0
        for k in range(0, len(norad_ids), IDS_PER_REQUEST):
            chunk = norad_ids[k:k + IDS_PER_REQUEST]
            written += write_by_satellite(query_gp_history(opener, chunk, start, end), tag)
            time.sleep(SECONDS_BETWEEN_REQUESTS)
        print(f'{root.name}: {written} satellites written for {start:%Y-%m-%d %H:%M} .. {end:%Y-%m-%d %H:%M} UTC')


if __name__ == '__main__':
    main(sys.argv[1:])
