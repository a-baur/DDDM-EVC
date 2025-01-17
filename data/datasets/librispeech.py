import os

import torchaudio


def load_librispeech(
    root: str, url: str, folder_in_archive: str
) -> torchaudio.datasets.LIBRISPEECH:
    root = os.path.join(root, "librispeech")
    if not os.path.exists(root):
        os.makedirs(root)
    ds = torchaudio.datasets.LIBRISPEECH(root, url, folder_in_archive, download=True)
    return ds


if __name__ == "__main__":
    load_librispeech(".", "dev-clean", "LibriSpeech")
