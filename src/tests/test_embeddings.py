import torch

import util
from config import Config
from data import AudioDataloader, MelSpectrogramFixed
from models import PitchEncoder, SpeakerEncoder
from util import get_root_path, get_yaapt_f0


def test_speaker_encoder(config: Config, dataloader: AudioDataloader) -> None:
    """Test Meta-StyleSpeech encoder."""
    speaker_encoder = SpeakerEncoder(config.models.speaker_encoder)

    ckpt_file = get_root_path() / "ckpt" / "speaker_encoder.pth"
    speaker_encoder.load_state_dict(
        torch.load(ckpt_file.as_posix(), map_location="cpu", weights_only=True)
    )

    mel_transform = MelSpectrogramFixed(config.data.mel_transform)

    batch = next(iter(dataloader))

    x: torch.Tensor
    length: torch.Tensor
    x, length = batch

    x_mel = mel_transform(x)  # B x C x T
    mask = util.sequence_mask(length, x_mel.size(2)).to(x_mel.dtype)  # B x T
    mask = mask.unsqueeze(1)  # B x 1 x T

    output = speaker_encoder(x_mel, mask)
    assert output.shape[0] == config.training.batch_size
    assert output.shape[1] == config.models.speaker_encoder.out_dim


def test_pitch_encoder(config: Config, dataloader: AudioDataloader) -> None:
    """Test VQ-VAE pitch encoder."""
    pitch_encoder = PitchEncoder(config.models.pitch_encoder)

    batch = next(iter(dataloader))

    x: torch.Tensor
    x, _ = batch

    f0 = get_yaapt_f0(x.numpy(), config.data.dataset.sampling_rate)
    ii = f0 != 0
    f0[ii] = (f0[ii] - f0[ii].mean()) / f0[ii].std()
    f0 = torch.FloatTensor(f0)  # (B x T/80)
    output = pitch_encoder.code_extraction(f0)

    assert output.shape[0] == config.training.batch_size
    assert output.min().item() == 0
    assert output.max().item() == config.models.pitch_encoder.vq.k_bins - 1
    assert not output.is_floating_point()  # Discrete codes
