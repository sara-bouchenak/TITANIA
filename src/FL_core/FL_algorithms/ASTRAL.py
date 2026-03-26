from typing import Iterable, Sequence
from torch.nn import Module
import itertools
import time

from scipy.optimize import differential_evolution

from fluke import FlukeENV
from fluke.client import Client
from fluke.server import Server
from fluke.data import FastDataLoader
from fluke.utils.model import aggregate_models

from src.FL_core.FL_algorithms import CustomCentralizedFL, CustomServer


class ASTRALServer(CustomServer):

    def __init__(
        self,
        model: Module,
        test_set: FastDataLoader | None,
        val_set:  FastDataLoader | None,
        clients: Sequence[Client],
        fairness_metric_name: str,
        sensitive_attributes: list[str],
        eps: float,
        maxiter: int,
        popsize: int,
        tol: float,
        mutation: list[float],
        recombination: float,
        workers: int,
        seed: int | None,
        weighted: bool = False,
        lr: float = 1,
        **kwargs,
    ):
        super().__init__(model, test_set, val_set, clients, weighted, lr, **kwargs)

        assert val_set != None

        self.fairness_metric_name = fairness_metric_name
        self.sensitive_attributes = sensitive_attributes
        self.eps = eps
        self.maxiter = maxiter
        self.popsize = popsize
        self.tol = tol
        self.mutation = mutation
        self.recombination = recombination
        self.workers = workers
        self.seed = seed

    def aggregate(self, eligible: Sequence[Client], client_models: Iterable[Module]) -> None:
        start_aggregate_time = time.time()
        client_models_copy, client_models = itertools.tee(client_models, 2)
        client_models_list = [client_model for client_model in client_models_copy]
        weights = self.get_client_weights_with_DE(client_models_list)
        aggregate_models(self.model, client_models, weights, self.hyper_params.lr, inplace=True)
        end_aggregate_time = time.time()

        aggregate_time = (end_aggregate_time - start_aggregate_time)

        self.notify(
                event="track_item",
                round=self.rounds+1,
                item="aggregate_time",
                value=aggregate_time,
            )

    def get_client_weights_with_DE(self, client_models: list[Module]) -> list[float]:

        bounds = [[-1, 1]]*len(client_models)

        print('[ASTRAL] Start')
        result = differential_evolution(
            self._compute_score,
            bounds=bounds,
            args=(client_models,),
            strategy='best1bin',
            maxiter=self.maxiter,
            popsize=self.popsize,
            tol=self.tol,
            disp=True,
            mutation=tuple(self.mutation),
            recombination=self.recombination,
            workers=self.workers,
            rng=self.seed,
            polish=False,
        )
        print('[ASTRAL] Status : %s' % result['message'])
        print('[ASTRAL] Total Evaluations: %d' % result['nfev'])

        weights = list(result['x'])
        normalized_weights = [w/sum(weights) for w in weights]
        print("[ASTRAL] Client weights: {}".format(normalized_weights))

        accuracy, bias_metrics = self._aggregate_and_evaluate(normalized_weights, client_models)
        print("[ASTRAL] Accuracy: {}".format(accuracy))
        print("[ASTRAL] Metrics {}: {}".format(self.fairness_metric_name, bias_metrics))

        for metric in bias_metrics:
            if abs(metric) > self.eps:
                #raise ValueError("[ASTRAL] Bias metrics is superior to epsilon")
                pass

        return normalized_weights

    def _compute_score(self, weights: list[float], client_models: list[Module]):
        normalized_weights = [w/sum(weights) for w in weights]
        accuracy, bias_metrics = self._aggregate_and_evaluate(normalized_weights, client_models)

        score = accuracy
        for metric in bias_metrics:
            if abs(metric) >= self.eps:
                score -= abs(metric)
        return -score

    def _aggregate_and_evaluate(self, normalized_weights: list[float], client_models: list[Module]):

        global_model = aggregate_models(self.model, iter(client_models), normalized_weights, self.hyper_params.lr, inplace=False)

        evaluator = FlukeENV().get_evaluator()
        evals = evaluator.evaluate(self.rounds + 1, global_model, self.val_set, loss_fn=None, device=self.device)
        del global_model

        bias_metrics = []
        for metric_name, metric_val in evals.items():
            if self.fairness_metric_name in metric_name:
                for sensitive_attribute in self.sensitive_attributes:
                    if sensitive_attribute in metric_name:
                        bias_metrics.append(metric_val)

        return evals["accuracy"], bias_metrics


class ASTRAL(CustomCentralizedFL):

    def get_server_class(self) -> type[Server]:
        return ASTRALServer
