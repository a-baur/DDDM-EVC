import torch

from config import load_hydra_config
from data import AudioDataloader, MelTransform, MSPPodcast
from models import dddm_from_config

torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = load_hydra_config("dddm_vc_xlsr")
mel_transform = MelTransform(config.data.mel_transform).to(device)
model = dddm_from_config(config.model, pretrained=False).to(device)
model = torch.compile(model)

dataset = MSPPodcast(config.data, split="development")
dl = AudioDataloader(
    dataset=dataset,
    cfg=config.data.dataloader,
    batch_size=config.training.batch_size,
)

if __name__ == "__main__":
    x, x_n_frames = next(iter(dl))
    x, x_n_frames = x.to(device), x_n_frames.to(device)
    x_mel = mel_transform(x)

    # model(x, x_mel, x_n_frames)
    # summary(model, input_data=(x, x_mel, x_n_frames))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.autograd.profiler.profile(
        profile_memory=True,
        record_shapes=True,
        use_device="cuda",
        with_stack=True,
        with_modules=True,
        with_flops=True,
    ) as prof:
        start.record()
        diff_loss, rec_loss = model.compute_loss(x, x_mel, x_n_frames)
        loss = diff_loss + rec_loss
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
