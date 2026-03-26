import time
from typing import Any, Sequence
from torch.nn import Module
from copy import deepcopy

from fluke import FlukeENV, DDict
from fluke.client import Client
from fluke.server import Server
from fluke.algorithms import CentralizedFL
from fluke.data import DataSplitter, FastDataLoader
from fluke.evaluation import Evaluator
from fluke.utils import get_loss, get_class_from_qualified_name
from fluke.config import OptimizerConfigurator
from torch.utils.data import DataLoader


class CustomClient(Client):

    def __init__(
        self,
        index: int,
        train_set: FastDataLoader | DataLoader,
        test_set: FastDataLoader | DataLoader,
        val_set: FastDataLoader | None,
        optimizer_cfg: OptimizerConfigurator,
        loss_fn: Module,
        local_epochs: int,
        fine_tuning_epochs: int = 0,
        clipping: float = 0,
        persistency: bool = True,
        **kwargs,
    ):
        super().__init__(index, train_set, test_set, optimizer_cfg, loss_fn, local_epochs, fine_tuning_epochs, clipping, persistency, **kwargs)
        self.val_set = val_set

    def evaluate(self, evaluator: Evaluator, test_set: FastDataLoader) -> dict[str, float]:
        model = self.model
        if test_set is not None and model is not None:
            evaluation = evaluator.evaluate(
                self._last_round, model, test_set, device=self.device, loss_fn=self.hyper_params.loss_fn
            )
            return evaluation
        return {}


class CustomServer(Server):

    def __init__(
        self,
        model: Module,
        test_set: FastDataLoader | None,
        val_set:  FastDataLoader | None,
        clients: Sequence[Client],
        weighted: bool = False,
        lr: float = 1,
        **kwargs,
    ):
        super().__init__(model, test_set, clients, weighted, lr, **kwargs)
        self.val_set = val_set

        if "time_to_accuracy_target" in kwargs.keys():
            self.time_to_accuracy_target = kwargs["time_to_accuracy_target"]
        else:
            self.time_to_accuracy_target = None

        if "loss" in kwargs.keys():
            self.loss_fn = get_loss(kwargs["loss"]) if isinstance(kwargs["loss"], str) else kwargs["loss"]()
        else:
            self.loss_fn = None

    def fit(
        self, n_rounds: int = 10,
        eligible_perc: float = 0.1,
        finalize: bool = True,
        **kwargs,
    ) -> None:

        self.start_fit_time = time.time()
        super().fit(n_rounds, eligible_perc, finalize, **kwargs)
        end_fit_time = time.time()

        training_time = (end_fit_time - self.start_fit_time)
        self.notify(
                event="track_item",
                round=-1,
                item="training_time",
                value=training_time,
            )

        training_time_per_participant_per_round = training_time / (self.n_clients * eligible_perc * n_rounds)
        self.notify(
                event="track_item",
                round=-1,
                item="training_time_per_participant_per_round",
                value=training_time_per_participant_per_round,
            )

    def evaluate(self, evaluator: Evaluator, test_set: FastDataLoader) -> dict[str, float]:

        if test_set is not None:
            metrics = evaluator.evaluate(
                self.rounds + 1, self.model, test_set, loss_fn=self.loss_fn, device=self.device
            )
        else:
            metrics = {}

        if self.time_to_accuracy_target is not None and "accuracy" in metrics.keys():

            if metrics["accuracy"] >= self.time_to_accuracy_target:
                self.time_to_accuracy_target = None
    
                target_accuracy_time = time.time()
                time_to_accuracy = target_accuracy_time - self.start_fit_time

                self.notify(
                        event="track_item",
                        round=-1,
                        item="time_to_accuracy",
                        value=time_to_accuracy,
                    )

        return metrics


class CustomCentralizedFL(CentralizedFL):

    def __init__(self, n_clients: int, data_splitter: DataSplitter, hyper_params: DDict, val_data: dict, **kwargs):
        self.clients_val = val_data["clients_val"]
        self.server_val = val_data["server_val"]

        # For allowing torchmetrics loss
        if "torchmetrics" in hyper_params["client"]["loss"]:
            hyper_params["client"]["loss"] = get_class_from_qualified_name(hyper_params["client"]["loss"])
        if "torchmetrics" in hyper_params["server"]["loss"]:
            hyper_params["server"]["loss"] = get_class_from_qualified_name(hyper_params["server"]["loss"])

        super().__init__(n_clients, data_splitter, hyper_params, **kwargs)

    def get_client_class(self) -> type[Client]:
        return CustomClient

    def get_server_class(self) -> type[Server]:
        return CustomServer

    def init_server(self, model: Any, data: FastDataLoader, config: DDict) -> Server:
        server: Server = self.get_server_class()(
            model=model, test_set=data, val_set=self.server_val, clients=self.clients, **config
        )
        if FlukeENV().get_save_options()[0] is not None:
            server.attach(self)
        return server

    def init_clients(
        self,
        clients_tr_data: list[FastDataLoader],
        clients_te_data: list[FastDataLoader],
        config: DDict,
    ) -> Sequence[Client]:
        self._fix_opt_cfg(config.optimizer)
        optimizer_cfg = OptimizerConfigurator(
            optimizer_cfg=config.optimizer, scheduler_cfg=config.scheduler
        )
        loss = get_loss(config.loss) if isinstance(config.loss, str) else config.loss()
        clients = [
            self.get_client_class()(
                index=i,
                train_set=clients_tr_data[i],
                test_set=clients_te_data[i],
                val_set=self.clients_val[i],
                optimizer_cfg=optimizer_cfg,
                loss_fn=deepcopy(loss),
                **config.exclude("optimizer", "loss", "batch_size", "scheduler"),
            )
            for i in range(self.n_clients)
        ]
        return clients
