import torch

from data import AudioDataloader, MSPPodcast
from config import load_hydra_config
from models import models_from_config
from util import get_root_path

cfg = load_hydra_config("evc_xlsr_yin", overrides=["data.dataset.segment_size=70000"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, preprocessor, style_encoder = models_from_config(cfg, device)

ckpt = torch.load(
    get_root_path() / "ckpt/evc_xlsr_yin.pth", map_location=device, weights_only=False
)

preprocessor.load_state_dict(ckpt["preprocessor"])
preprocessor.requires_grad_(False)
preprocessor.eval()
preprocessor.to(device)

style_encoder.load_state_dict(ckpt["style_encoder"])
style_encoder.requires_grad_(False)
style_encoder.eval()
style_encoder.to(device)

print("loaded models")


def avg_embed(filter_dim: str, value: float) -> None:
    dataset = MSPPodcast(
        cfg.data,
        split="development",
        random_segmentation=True,
        load_labels=True,
        label_filter={filter_dim: value},
    )
    dataloader = AudioDataloader(
        dataset,
        cfg=cfg.data.dataloader,
        batch_size=100,
        shuffle=True,
    )

    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    spk_embeds = []
    emo_embeds = []

    for idx, batch in enumerate(dataloader):
        print("\rprocessing batch", idx + 1, "/", len(dataloader), end="")
        audio, n_frames, labels = batch
        audio, n_frames, labels = (
            audio.to(device),
            n_frames.to(device),
            labels.to(device),
        )

        x = preprocessor(audio, n_frames, labels)

        spk = style_encoder.speaker_encoder(x)
        spk_embeds.append(spk)

        emo = style_encoder.emotion_encoder(x.audio, embeddings_only=True)
        emo_embeds.append(emo)

    spk_embeds = torch.cat(spk_embeds)
    emo_embeds = torch.cat(emo_embeds)

    avg_spk_embeds = spk_embeds.mean(-1)
    avg_emo_embeds = emo_embeds.mean(-1)

    f_emo_path = get_root_path() / "emo" / filter_dim / f"{value}.pt"
    f_emo_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(avg_spk_embeds, f_emo_path)

    f_spk_path = get_root_path() / "spk" / filter_dim / f"{value}.pt"
    f_spk_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(avg_emo_embeds, f_spk_path)


filter_dim = "EmoAct"
values = [1, 2, 3, 4, 5, 6, 7]

if __name__ == "__main__":
    for v in values:
        avg_embed(filter_dim, v)
