import os
import pathlib

import torchaudio


def _find_project_root() -> str:
    """Returns the root directory of the project"""
    current_dir = pathlib.Path(__file__).parent
    while not (current_dir / "pyproject.toml").exists():
        current_dir = current_dir.parent
    return str(current_dir)


def load_librispeech(
    url: str,
    folder_in_archive: str,
    path: str = _find_project_root(),
) -> torchaudio.datasets.LIBRISPEECH:
    """
    Load the LibriSpeech dataset.

    :param url: URL of the dataset
    :param folder_in_archive: Folder in the archive
    :param path: Path to the root directory (default: project root)
    :return: LibriSpeech dataset
    """
    root = os.path.join(path, "datasets/librispeech")
    if not os.path.exists(root):
        os.makedirs(root)
    ds = torchaudio.datasets.LIBRISPEECH(root, url, folder_in_archive, download=True)
    return ds


if __name__ == "__main__":
    load_librispeech(".", "dev-clean", "LibriSpeech")
