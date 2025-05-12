import torch

import util
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
    batch_size=100,
    shuffle=False,
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

util.load_model(
    style_encoder.speaker_encoder,
    "metastylespeech.pth",
    mode="eval",
    freeze=True,
)

optimizer = torch.optim.AdamW(
    style_encoder.parameters(),
    5e-5,
    betas=(0.85, 0.98),
    eps=1e-8,
)
scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.9999, last_epoch=-1
)

LOG_INTERVAL = 10
EVAL_INTERVAL = 50


def train(batch):
    style_encoder.train()
    optimizer.zero_grad()

    audio, n_frames, labels = (
        batch[0].to(device),
        batch[1].to(device),
        batch[2].to(device),
    )
    x = preprocessor(audio, n_frames, labels)
    loss, loss_spk, loss_emo, loss_spk_adv, loss_emo_adv = style_encoder.compute_loss(
        x, adv_loss_coef=0.1
    )
    return loss, loss_spk, loss_emo, loss_spk_adv, loss_emo_adv


eval_batch = next(iter(eval_dataloader))


def eval():
    style_encoder.eval()
    with torch.no_grad():
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
        ) = style_encoder.compute_loss(x_eval, adv_loss_coef=0.1)
        print(
            f">>> EVAL BATCH: "
            f"loss: {loss_eval.item():.4f}, loss_spk: {loss_spk_eval.item():.4f}, "
            f"loss_emo: {loss_emo_eval.item():.4f}, loss_spk_adv: {loss_spk_adv_eval.item():.4f}, "
            f"loss_emo_adv: {loss_emo_adv_eval.item():.4f}"
        )


def main():
    print("Training Style Encoder")
    preprocessor.eval()
    for i, batch in enumerate(train_dataloader):
        loss, loss_spk, loss_emo, loss_spk_adv, loss_emo_adv = train(batch)

        if i % LOG_INTERVAL == 0:
            print(
                f"BATCH {i}: "
                f"loss: {loss.item():.4f}, loss_spk: {loss_spk.item():.4f}, "
                f"loss_emo: {loss_emo.item():.4f}, loss_spk_adv: {loss_spk_adv.item():.4f}, "
                f"loss_emo_adv: {loss_emo_adv.item():.4f}"
            )
        if i % EVAL_INTERVAL == 0 and i > 0:
            eval()

        loss.backward()
        optimizer.step()
        scheduler_g.step()


if __name__ == "__main__":
    main()
