from src.TITANIA.data_cleaning.outliers.default import OutliersDataCleaningMethod
from src.TITANIA.data_cleaning.outliers.local_methods import LocalOLDataCleaning
from src.TITANIA.data_cleaning.outliers.global_methods import GlobalOLDataCleaning


class OutliersDataCleaningMethods():

    @classmethod
    def get(cls, name: str, global_stat_dict, **kwargs) -> OutliersDataCleaningMethod:
        if name not in _OL_DATA_CLEANING_METHODS_MAP:
            raise ValueError(
                f"Outliers data cleaning method {name} not found. The supported outliers data cleaning methods are: "
                + ", ".join(_OL_DATA_CLEANING_METHODS_MAP.keys())
                + "."
            )
        
        if name == "global":
            return _OL_DATA_CLEANING_METHODS_MAP[name](global_stat_dict=global_stat_dict, **kwargs)
        else:
            return _OL_DATA_CLEANING_METHODS_MAP[name](**kwargs)


_OL_DATA_CLEANING_METHODS_MAP = {
    "default": OutliersDataCleaningMethod,
    "local": LocalOLDataCleaning,
    "global": GlobalOLDataCleaning,
}
