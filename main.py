import os
import yaml
import typer
from rich.console import Console
import hydra
from omegaconf import DictConfig, OmegaConf

from fluke import FlukeENV, DDict
from fluke.utils import get_class_from_qualified_name
from fluke.utils.log import get_logger

from src.FL_core.fairness_eval.evaluators import BinaryClassificationFairnessEval
from src.FL_core.utils.configs import CustomConfiguration
from src.FL_core.data_loading.pipeline import data_loading_pipeline

console = Console()
app = typer.Typer()

OmegaConf.register_new_resolver("sanitize_override_dirname", lambda x: x.replace(os.path.sep, "_"))
OmegaConf.register_new_resolver("keep_last_str", lambda x: x.split(".")[-1])


def titania_run_federation(cfg: DDict, resume: str | None = None) -> None:

    data_splitter, val_data = data_loading_pipeline(cfg)

    FlukeENV().configure(cfg)

    # Automatically adjust some hyperparameters in cfg
    input_size = data_splitter.data_container.clients_tr[0].tensors[0].shape[-1]
    cfg.method.hyperparameters.net_args.input_size = input_size
    if data_splitter.data_container.num_classes <= 2:
        cfg.method.hyperparameters.net_args.num_classes = 1  
    else:
        cfg.method.hyperparameters.net_args.num_classes = data_splitter.data_container.num_classes

    # Save config file
    config_path = os.path.join(cfg.paths.output_dir, "config.yaml")
    cfg_to_save = cfg.to_dict()
    cfg_to_save["paths"]["output_dir"] = "${hydra:runtime.output_dir}"
    cfg_to_save["logger"]["json_log_dir"] = "${paths.output_dir}"
    yaml.dump(cfg_to_save, open(config_path, "w"))

    if cfg.exp.train:

        fl_algo_class = get_class_from_qualified_name(cfg.method.name)
        fl_algo = fl_algo_class(cfg.protocol.n_clients, data_splitter, cfg.method.hyperparameters, val_data=val_data)

        log_name = f"{fl_algo.__class__.__name__} [{fl_algo.id}]"
        log = get_logger(cfg.logger.name, name=log_name, **cfg.logger.exclude("name", "json_log_dir"))
        log.init(**cfg, exp_id=fl_algo.id)
        fl_algo.set_callbacks([log])
        FlukeENV().set_logger(log)

        evaluator = BinaryClassificationFairnessEval(
            eval_every=cfg.eval.eval_every, 
            n_classes=data_splitter.data_container.num_classes,
            sensitive_attributes=cfg.data.dataset.sensitive_attributes,
        )
        FlukeENV().set_evaluator(evaluator)

        if resume is not None:
            fl_algo.load(resume)
        try:
            fl_algo.run(cfg.protocol.n_rounds, cfg.protocol.eligible_perc)
        except Exception as e:
            log.log(f"Error: {e}")
            FlukeENV().force_close()
            FlukeENV.clear()
            log.close()
            FlukeENV().close_cache()
            raise e

        if cfg.logger.json_log_dir:
            results_path = os.path.join(cfg.logger.json_log_dir, "results.json")
            log.save(results_path)

        log.close()

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg : DictConfig) -> None:
    custom_cfg = CustomConfiguration(cfg)
    titania_run_federation(custom_cfg)


if __name__ == "__main__":
    main()
