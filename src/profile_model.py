import torch
from torch.profiler import ProfilerActivity, profile

from config import load_hydra_config
from data import AudioDataloader, MSPPodcastFilenames
from models import DDDM, models_from_config
from models.dddm.preprocessor import DDDMInput

# The flags below controls whether to allow TF32 on cuDNN. This flag defaults to True.
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = load_hydra_config("vc_xlsr")
model, preprocessor, style_encoder = models_from_config(config, device=device)
model.to(device)
preprocessor.to(device)
style_encoder.to(device)

model: DDDM = torch.compile(model, backend="inductor")

# dataset = MSPPodcast(config.data, split="development")
# collate_fn = None

dataset = MSPPodcastFilenames(config.data, split="development")

dl = AudioDataloader(
    dataset=dataset,
    cfg=config.data.dataloader,
    batch_size=config.training.batch_size,
)


if __name__ == "__main__":
    # model(x, x_mel, x_n_frames)
    # summary(model, input_data=(x, x_mel, x_n_frames))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
        with_flops=True,
    ) as prof:
        start.record()

        audio, n_frames, labels = next(iter(dl))
        audio, n_frames, labels = (
            audio.to(device),
            n_frames.to(device),
            labels.to(device),
        )
        x: DDDMInput = preprocessor(audio)
        g = style_encoder(x).unsqueeze(-1)
        score_loss, src_ftr_loss, rec_loss = model.compute_loss(x, g)
        loss = score_loss + src_ftr_loss + rec_loss
        loss.backward()

        end.record()

    torch.cuda.synchronize()
    print(f"Total runtime: {start.elapsed_time(end) / 1000}s")
    print(
        prof.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=10, top_level_events_only=False
        )
    )

    print("======================================")
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=10, top_level_events_only=False
        )
    )
