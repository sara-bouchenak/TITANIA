from torch.nn import Module

from fluke.utils.log import Log


class CustomLog(Log):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def end_fit(self, round: int, client_id: int, model: Module, loss: float, **kwargs) -> None:
        loss_dict = {"training_loss": loss}
        self.tracker.add(perf_type="post-fit", metrics=loss_dict, round=round, client_id=client_id)
        return super().end_fit(round, client_id, model, loss, **kwargs)
