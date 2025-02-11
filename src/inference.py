import hydra
from omegaconf import DictConfig, OmegaConf

import config

config.register_configs()


@hydra.main(
    version_base=None,
    config_path=config.CONFIG_PATH.as_posix(),
    config_name="vc_config",
)  # type: ignore
def inference(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    inference()
