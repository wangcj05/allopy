import pickle
from base64 import b64encode
from pathlib import Path
from typing import Optional

import requests

from allopy import OptData

_files = {
    "active-baseline": "https://1drv.ms/u/s!ArHJMsJpCjvrh9t43VV7HGEn5p7ZYw?e=8pW5UA",
    "active-cvar": "https://1drv.ms/u/s!ArHJMsJpCjvrh9t5naQroQGZghJgDQ?e=4maPuO",
    "active-downside": "https://1drv.ms/u/s!ArHJMsJpCjvrh9t6JqG3Q1rX411AJA",
    "active-upside": "https://1drv.ms/u/s!ArHJMsJpCjvrh9t7uSepNSzmbVHKng"
}


def fetch_opt_data_test_file(name: str) -> Optional[OptData]:
    name = name.lower()
    if name not in _files:
        raise ValueError(f"Unknown file: {name}")

    p = Path.home().joinpath(".allopy", "test-data")
    if not p.exists():
        p.mkdir(777, True, True)

    fp = p.joinpath(name)
    if not fp.exists():
        ok = download_from_1drv(_files[name], fp.as_posix())
        if not ok:
            return None

    with open(fp.as_posix(), 'rb') as f:
        return pickle.load(f)


def download_from_1drv(src_url: str, save_file_path: str):
    payload = b64encode(src_url.encode()).decode().strip('=').replace('/', '_')
    r = requests.get(f"https://api.onedrive.com/v1.0/shares/u!{payload}/root?expand=children")

    if r.status_code != 200:
        raise ConnectionError("could not retrieve meta data for file")

    link = r.json().get('@content.downloadUrl', None)
    if link is None:
        raise ValueError("download url not available in file meta data")

    r = requests.get(link)
    if r.status_code != 200:
        return False

    with open(save_file_path, 'wb') as f:
        f.write(r.content)

    return True
