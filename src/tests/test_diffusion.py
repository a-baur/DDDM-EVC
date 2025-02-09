import torch

from config import Config
from data import AudioDataloader
from models import Diffusion
from util import get_root_path, load_model, pad_tensors_to_length, sequence_mask
from util.sequences import get_u_net_compatible_length


def test_diffusion(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test diffusion model"""
    # Load test data
    testdata = torch.load(
        get_root_path() / "src" / "tests" / "testdata" / "testdata.pth"
    )
    x_mel = testdata["x_mel"]
    src_out, ftr_out = testdata["src_mel"], testdata["ftr_mel"]
    spk = testdata["spk_emb"]
    x_lengths = torch.LongTensor([x_mel.size(-1)])
    x_mask = sequence_mask(x_lengths, x_mel.size(2)).to(x_mel.dtype)

    # Load diffusion model
    diffusion = Diffusion(cfg.models.diffusion)
    load_model(diffusion, "diffusion.pth", freeze=True)

    src_mean_x, ftr_mean_x = diffusion.compute_diffused_mean(
        x_mel, x_mask, src_out, ftr_out, 1.0
    )

    # Get the U-Net compatible length
    max_length = int(x_lengths.max())
    max_length_new = get_u_net_compatible_length(max_length)

    # Pad the sequences for U-Net compatibility
    src_mean_x, ftr_mean_x, src_out, ftr_out = pad_tensors_to_length(
        [src_mean_x, ftr_mean_x, src_out, ftr_out], max_length_new
    )

    if max_length_new > max_length:
        x_mask = sequence_mask(x_lengths, max_length_new).to(x_mel.dtype)

    # Add noise to diffused mean to create priors for diffusion
    start_n = torch.randn_like(src_mean_x, device=src_mean_x.device)
    src_mean_x.add_(start_n)
    ftr_mean_x.add_(start_n)

    # Diffusion
    y_src, y_ftr = diffusion(
        src_mean_x, ftr_mean_x, x_mask, src_out, ftr_out, spk, 6, "ml"
    )
    y = (y_src + y_ftr) / 2
    enc_out = src_out + ftr_out
    torch.save(y, "./testdata/diffusion_test.pth")
    torch.save(enc_out, "./testdata/diffusion_enc_out.pth")
