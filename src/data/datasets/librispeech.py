import pathlib

import torchaudio

from util import get_root_path


def load_librispeech(
    url: str,
    folder_in_archive: str,
    path: str | pathlib.Path = get_root_path(),
) -> torchaudio.datasets.LIBRISPEECH:
    """
    Load the LibriSpeech dataset.

    :param url: URL of the dataset
    :param folder_in_archive: Folder in the archive
    :param path: Path to the root directory (default: project root)
    :return: LibriSpeech dataset
    """
    path = pathlib.Path(path)
    root = path / "datasets" / "librispeech"
    if not root.exists():
        root.mkdir(parents=True)
    ds = torchaudio.datasets.LIBRISPEECH(
        root.as_posix(), url, folder_in_archive, download=True
    )
    return ds


if __name__ == "__main__":
    load_librispeech(".", "dev-clean", "LibriSpeech")
