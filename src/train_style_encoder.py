"""
Minimal train script for optimizing projectin layers of the style encoder
to maximize the disentanglment of the speaker and emotion embeddings.

Approach:
Minimize speaker classification loss of speaker encoder while maximizing
emotion regression loss of speaker encoder via Gradient Reversal Layer.
Vice versa for the emotion encoder.
"""

import torch

from config import load_hydra_config
from data import AudioDataloader, MelTransform, MSPPodcast
from models import DDDMPreprocessor
from models.content_encoder import XLSR_ESPEAK_CTC
from models.pitch_encoder import YINEncoder
from models.style_encoder import DisentangledStyleEncoder

cfg = load_hydra_config("evc_xlsr_yin", overrides=["data.dataset.segment_size=38000"])
train_dataloader = AudioDataloader(
    dataset=MSPPodcast(cfg.data, split="train", random_segmentation=True),
    cfg=cfg.data.dataloader,
    batch_size=32,
    shuffle=True,
)
eval_dataloader = AudioDataloader(
    dataset=MSPPodcast(cfg.data, split="test1", random_segmentation=True),
    cfg=cfg.data.dataloader,
    batch_size=32,
    shuffle=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocessor = DDDMPreprocessor(
    mel_transform=MelTransform(cfg.data.mel_transform).to(device),
    pitch_encoder=YINEncoder(cfg.model.pitch_encoder).to(device),
    content_encoder=XLSR_ESPEAK_CTC(
        return_logits=False,
        return_hidden=True,
    ),
    sample_rate=cfg.data.dataset.sampling_rate,
).to(device)

style_encoder = DisentangledStyleEncoder(
    cfg.model.style_encoder,
    hidden_dim=256,
    n_spk=1459,
).to(device)


spk_params = list(style_encoder.spk_proj.parameters()) + list(
    style_encoder.spk_cls.parameters()
)
optimizer_spk = torch.optim.AdamW(
    spk_params,
    lr=1e-6,
    weight_decay=0.0,
)

emo_params = (
    list(style_encoder.emo_proj.parameters())
    + list(style_encoder.emo_reg.parameters())
    + list(style_encoder.emo_adv.parameters())
    + list(style_encoder.spk_adv.parameters())
)
optimizer_emo = torch.optim.AdamW(
    emo_params,
    lr=1e-5,
    weight_decay=1e-4,
)

LOG_INTERVAL = 10
EVAL_INTERVAL = 50
EVAL_BATCHES = 10
CKPT_INTERVAL = 1950  # 1 epoch = 1950 batches


def train(batch):
    style_encoder.train()
    optimizer_spk.zero_grad()
    optimizer_emo.zero_grad()

    audio, n_frames, labels = (
        batch[0].to(device),
        batch[1].to(device),
        batch[2].to(device),
    )
    x = preprocessor(audio, n_frames, labels)
    loss, loss_spk, loss_emo, loss_spk_adv, loss_emo_adv = style_encoder.compute_loss(
        x, adv_spk_coef=0.0, adv_emo_coef=0.1
    )
    return loss, loss_spk, loss_emo, loss_spk_adv, loss_emo_adv


@torch.no_grad()
def eval():
    style_encoder.eval()

    losses = {
        "loss": 0.0,
        "loss_adv": 0.0,
        "loss_spk": 0.0,
        "loss_emo": 0.0,
        "loss_spk_adv": 0.0,
        "loss_emo_adv": 0.0,
    }

    eval_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    for i, eval_batch in enumerate(eval_dataloader):
        if i >= EVAL_BATCHES:
            break
        eval_audio, eval_n_frames, eval_labels = (
            eval_batch[0].to(device),
            eval_batch[1].to(device),
            eval_batch[2].to(device),
        )
        x_eval = preprocessor(eval_audio, eval_n_frames, eval_labels)
        (
            loss_eval,
            loss_spk_eval,
            loss_emo_eval,
            loss_spk_adv_eval,
            loss_emo_adv_eval,
        ) = style_encoder.compute_loss(
            x_eval, adv_spk_coef=0.0, adv_emo_coef=0.1, include_acc=True
        )
        loss = loss_spk_eval.item() + loss_emo_eval.item()
        adv_loss = loss_emo_adv_eval.item() + loss_spk_adv_eval.item()
        losses["loss"] += loss
        losses["loss_adv"] += adv_loss
        losses["loss_spk"] += loss_spk_eval.item()
        losses["loss_emo"] += loss_emo_eval.item()
        losses["loss_spk_adv"] += loss_spk_adv_eval.item()
        losses["loss_emo_adv"] += loss_emo_adv_eval.item()

    losses = {k: v / EVAL_BATCHES for k, v in losses.items()}

    print(
        f">>> EVAL BATCH: "
        f"loss: {losses['loss']:.4f}, "
        f"loss_adv: {losses['loss_adv']:.4f}, "
        f"loss_spk: {losses['loss_spk']:.4f}, "
        f"loss_emo: {losses['loss_emo']:.4f}, "
        f"loss_spk_adv: {losses['loss_spk_adv']:.4f}, "
        f"loss_emo_adv: {losses['loss_emo_adv']:.4f}"
    )


def main():
    print("Training Style Encoder")
    preprocessor.eval()
    for j in range(0, 100):
        for i, batch in enumerate(train_dataloader):
            loss, loss_spk, loss_emo, loss_spk_adv, loss_emo_adv = train(batch)

            if i % LOG_INTERVAL == 0:
                print(
                    f"EPOCH {j} BATCH {i}: "
                    f"loss: {loss.item():.4f}, loss_spk: {loss_spk.item():.4f}, "
                    f"loss_emo: {loss_emo.item():.4f}, loss_spk_adv: {loss_spk_adv.item():.4f}, "
                    f"loss_emo_adv: {loss_emo_adv.item():.4f}"
                )
            if i % EVAL_INTERVAL == 0 and i > 0:
                eval()
            if i % CKPT_INTERVAL == 0 and i > 0:
                print("Saving checkpoint...")
                torch.save(
                    {
                        "epoch": j,
                        "batch": i,
                        "model": style_encoder.state_dict(),
                    },
                    f"style_encoder_{j}_{i}.pth",
                )

            loss.backward()

            optimizer_spk.step()
            optimizer_emo.step()


if __name__ == "__main__":
    main()
