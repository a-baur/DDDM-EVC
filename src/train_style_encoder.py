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
    batch_size=100,
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

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, style_encoder.parameters()),
    lr=1e-5,
    weight_decay=1e-4,
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
        x, adv_spk_coef=0.001, adv_emo_coef=0.1
    )
    return loss, loss_spk, loss_emo, loss_spk_adv, loss_emo_adv


dl_iter = iter(eval_dataloader)


def eval():
    style_encoder.eval()
    with torch.no_grad():
        eval_batch = next(dl_iter)
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
            x_eval, adv_spk_coef=0.000001, adv_emo_coef=0.1, include_acc=True
        )
        loss = loss_spk_eval.item() + loss_emo_eval.item()
        adv_loss = loss_emo_adv_eval.item() + loss_spk_adv_eval.item()
        print(
            f">>> EVAL BATCH: "
            f"loss: {loss:.4f} | adv loss: {adv_loss:.4f} | loss_spk: {loss_spk_eval.item():.4f} | "
            f"loss_emo: {loss_emo_eval.item():.4f} | loss_spk_adv: {loss_spk_adv_eval.item():.4f}"
            f" | loss_emo_adv: {loss_emo_adv_eval.item():.4f}"
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

            loss.backward()
            optimizer.step()
            scheduler_g.step()


if __name__ == "__main__":
    main()
