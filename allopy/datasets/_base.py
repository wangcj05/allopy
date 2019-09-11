import pathlib

import requests


def filepath(fn: str, download: bool):
    return FileHandler(fn, download).filepath


class FileHandler:
    data_home = pathlib.Path.home().joinpath(".allopy", "data")

    def __init__(self, name: str, download: bool):
        self.name = name

        if not self.data_home.exists():
            self.data_home.mkdir(777, True, exist_ok=True)

        self._fp = self.data_home.joinpath(name)
        if not self._fp.exists() or download:
            self._download_file()

    @property
    def filepath(self):
        return self._fp.as_posix()

    @property
    def file_url(self):
        return f"https://github.com/DanielBok/allopy/blob/master/allopy/datasets/data/{self.name}?raw=true"

    def _download_file(self):
        print("Downloading required file from source")

        with requests.get(self.file_url) as r:
            with open(self.filepath, 'wb') as f:
                f.write(r.content)

        print("Download complete")
