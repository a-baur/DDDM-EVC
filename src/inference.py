import argparse
from pathlib import Path

import torch
import torchaudio

import config
import util
from models import HifiGAN, models_from_config


def inference(
    source_path: str,
    target_path: str,
    output_path: str,
    config_name: str,
    n_timesteps: int,
) -> None:
    print("Instantiating models...")
    outpath = Path(output_path)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    cfg = config.load_hydra_config(config_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_audio, src_sr = torchaudio.load(source_path)
    tgt_audio, tgt_sr = torchaudio.load(target_path)
    assert src_sr == tgt_sr, "Source and target audio must have the same sampling rate"

    model, preprocessor, style_encoder = models_from_config(
        cfg, sample_rate=src_sr, device=device
    )

    vocoder = HifiGAN(cfg.model.vocoder)
    util.load_model(vocoder, "hifigan.pth", freeze=True)

    src_audio, tgt_audio, vocoder = util.move_to_device(
        (src_audio, tgt_audio, vocoder),
        device,
    )

    print("Pre-processing input...")
    x = preprocessor(src_audio)
    t = preprocessor(tgt_audio)

    print("Generating...")
    g = style_encoder(t).unsqueeze(-1)
    y_mel = model(x, g, n_timesteps)

    print("Vocoding...")
    y = vocoder(y_mel)
    y = y.squeeze(1).detach().cpu()

    torchaudio.save(outpath, y, cfg.data.dataset.sampling_rate, encoding="PCM_F")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--source_path",
        type=str,
        default="./sample/src.wav",
        help="Path to the input audio wav",
    )
    parser.add_argument(
        "-t",
        "--target_path",
        type=str,
        default="./sample/trg.wav",
        help="Path to the conversion target audio wav",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="./sample/out.wav",
        help="Dir for output wav",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="dddm_vc_xlsr",
        help="Name of config to use",
    )
    parser.add_argument(
        "-n",
        "--n_timesteps",
        type=int,
        default=6,
        help="Number of diffusion timesteps",
    )

    args = parser.parse_args()
    inference(
        source_path=args.source_path,
        target_path=args.target_path,
        output_path=args.output_path,
        config_name=args.config,
        n_timesteps=args.n_timesteps,
    )
