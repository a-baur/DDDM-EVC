import argparse
from pathlib import Path

import torch
import torchaudio

import config
import util
from data import MelTransform
from models import DDDM, HifiGAN

config.register_configs()


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

    cfg = config.load_hydra_config_vc(config_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mel_transform = MelTransform(cfg.data.mel_transform)
    model = DDDM.from_config(cfg, pretrained=True)
    vocoder = HifiGAN(cfg.model.vocoder)
    util.load_model(vocoder, "hifigan.pth", freeze=True)

    print("Pre-processing input...")
    x, _ = torchaudio.load(source_path)
    x_mel = mel_transform(x)
    x_n_frames = torch.Tensor([x_mel.size(-1)])

    t, _ = torchaudio.load(target_path)
    t_mel = mel_transform(t)
    t_n_frames = torch.Tensor([t_mel.size(-1)])

    x, x_mel, x_n_frames, t, t_mel, t_n_frames, model, vocoder = util.move_to_device(
        (x, x_mel, x_n_frames, t, t_mel, t_n_frames, model, vocoder), device
    )

    print("Generating...")
    y_mel = model.voice_conversion(
        x, x_mel, x_n_frames, t, t_mel, t_n_frames, n_timesteps
    )

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
        default="config_vc.yaml",
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
