from src.TITANIA.data_cleaning.missing_values.default import MissingValuesDataCleaningMethod
from src.TITANIA.data_cleaning.missing_values.local_methods import LocalMVDataCleaning
from src.TITANIA.data_cleaning.missing_values.global_methods import GlobalMVDataCleaning
from src.TITANIA.data_cleaning.missing_values.Cafe import Cafe


class MissingValuesDataCleaningMethods():

    @classmethod
    def get(cls, name: str, global_stat_dict, **kwargs) -> MissingValuesDataCleaningMethod:
        if name not in _MV_DATA_CLEANING_METHODS_MAP:
            raise ValueError(
                f"Missing values data cleaning method {name} not found. The supported missing values data cleaning methods are: "
                + ", ".join(_MV_DATA_CLEANING_METHODS_MAP.keys())
                + "."
            )

        if name == "global":
            return _MV_DATA_CLEANING_METHODS_MAP[name](global_stat_dict=global_stat_dict, **kwargs)
        else:
            return _MV_DATA_CLEANING_METHODS_MAP[name](**kwargs)


_MV_DATA_CLEANING_METHODS_MAP = {
    "default": MissingValuesDataCleaningMethod,
    "local": LocalMVDataCleaning,
    "global": GlobalMVDataCleaning,
    "Cafe": Cafe
}
