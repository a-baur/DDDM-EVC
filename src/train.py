from config.schema import Config

if __name__ == "__main__":
    cfg = Config.from_yaml("config/config.yaml")
    print(cfg.dataloader.num_workers)
