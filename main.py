import traceback
import hydra
from omegaconf import DictConfig
from src.utils.global_config import set_config
from src.train import train_model
from src.test import test_model
from src.data.preprocessing import generate_training_data_dir,split_dataset

@hydra.main(config_path="config", version_base=None)
def main(cfg: DictConfig):
    try:
        set_config(cfg)
        mode = cfg.get("mode", "train")
        if mode == "train":
            train_model()
        elif mode == "test":
            test_model()
        elif mode == "data_gen":
            cf=cfg.preprocessing
            generate_training_data_dir(cf.raw_dataset_dir,cf.training_file_dir,cf.point_cloud_size)
        elif mode == "split_dataset":
            cf=cfg.preprocessing
            split_dataset(cf.training_file_dir,cf.filelist_dir,cf.split_ratio)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()