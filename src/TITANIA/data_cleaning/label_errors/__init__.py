from src.TITANIA.data_cleaning.label_errors.default import LabelErrorsDataCleaningMethod
from src.TITANIA.data_cleaning.label_errors.local_methods import LocalLEDataCleaning
from src.TITANIA.data_cleaning.label_errors.FedCorr import FedCorr


class LabelErrorsDataCleaningMethods():

    @classmethod
    def get(cls, name: str, **kwargs) -> LabelErrorsDataCleaningMethod:
        if name not in _LE_DATA_CLEANING_METHODS_MAP:
            raise ValueError(
                f"Label errors data cleaning method {name} not found. The supported label errors data cleaning methods are: "
                + ", ".join(_LE_DATA_CLEANING_METHODS_MAP.keys())
                + "."
            )

        return _LE_DATA_CLEANING_METHODS_MAP[name](**kwargs)


_LE_DATA_CLEANING_METHODS_MAP = {
    "default": LabelErrorsDataCleaningMethod,
    "local": LocalLEDataCleaning,
    "FedCorr": FedCorr,
}
