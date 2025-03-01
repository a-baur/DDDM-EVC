from pathlib import Path

import torch

from models import DDDMBatchInput


def test_batch_save_load(tmp_path: Path) -> None:
    batch_size = 10
    batch = DDDMBatchInput(
        audio=torch.rand(batch_size, 100),
        mel=torch.rand(batch_size, 80, 100),
        mask=torch.rand(batch_size, 80, 100),
        emb_pitch=torch.rand(batch_size, 100),
        emb_content=torch.rand(batch_size, 100),
    )
    filenames = [f"sample_{i}.pt" for i in range(batch_size)]

    batch.save(tmp_path, filenames)
    loaded = DDDMBatchInput.load(tmp_path, filenames)

    assert batch.audio.eq(loaded.audio).all()
    assert batch.mel.eq(loaded.mel).all()
    assert batch.mask.eq(loaded.mask).all()
    assert batch.emb_pitch.eq(loaded.emb_pitch).all()
    assert batch.emb_content.eq(loaded.emb_content).all()
