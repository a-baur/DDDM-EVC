import torch
from torch.profiler import ProfilerActivity, profile

from config import load_hydra_config
from data import AudioDataloader, MelTransform, MSPPodcast
from models import DDDM, dddm_from_config

# The flags below controls whether to allow TF32 on cuDNN. This flag defaults to True.
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = load_hydra_config("dddm_vc_xlsr")
mel_transform = MelTransform(config.data.mel_transform).to(device)
model = dddm_from_config(config.model, pretrained=False).to(device)
model: DDDM = torch.compile(model, backend="inductor")

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
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
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
