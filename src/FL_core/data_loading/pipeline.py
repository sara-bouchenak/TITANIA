import pandas as pd
import os
import yaml
import json
import random
import numpy as np

from fluke.data import DummyDataContainer

from src.FL_core.data_loading.sensitive_datasets import load_sensitive_dataset
from src.FL_core.data_loading.data_splitter import CustomDataSplitter, DummyDataSplitter
from src.TITANIA.data_cleaning.pipeline import clean_data
from src.FL_core.data_loading.data_processing import one_hot_encoding, convert_bool_and_cat_to_num, normalization, final_data_preprocessing
from src.TITANIA.noise_injection.add_noise import inject_noise


def data_loading_pipeline(cfg):

    np.random.seed(cfg.data.seed)
    random.seed(cfg.data.seed)

    if ("static" in cfg.data.loading.keys()) and cfg.data.loading.static:

        splitted_data, is_data_cleaned = load_data_from_folder(cfg.to_dict()["data"], cfg.protocol.n_clients)

    else:

        data = load_sensitive_dataset(
            dataset_name=cfg.data.dataset.dataset_name,
            **cfg.data.dataset.exclude("dataset_name")
        )

        data_splitter = CustomDataSplitter(
            data_dict=data,
            distribution=cfg.data.distribution.name,
            dist_args=cfg.data.distribution.exclude("name"),
            **cfg.data.others,
        )

        splitted_data = data_splitter.assign(cfg.protocol.n_clients)
        is_data_cleaned = False

    if not is_data_cleaned:

        if ("save_data_before_cleaning" in cfg.data.loading.keys()) and (cfg.data.loading.save_data_before_cleaning):
            save_data_to_pkl(splitted_data, cfg.to_dict()["data"], mode="before_cleaning")

        if "noise_injection" in cfg.data.keys():
            noisy_data = inject_noise(data=splitted_data, cfg=cfg.data.noise_injection, sensitive_attributes=cfg.data.dataset.sensitive_attributes)
        else:
            noisy_data = splitted_data

        cleaned_data, data_cleaning_metrics = clean_data(noisy_data, cfg=cfg.data.cleaning, sensitive_attributes=cfg.data.dataset.sensitive_attributes)

        data_cleaning_metrics_path = os.path.join(cfg.paths.output_dir, "data_cleaning_metrics.json")
        json.dump(data_cleaning_metrics, open(data_cleaning_metrics_path, "w"), indent=4)

        if ("save_data_after_cleaning" in cfg.data.loading.keys()) and (cfg.data.loading.save_data_after_cleaning):
            save_data_to_pkl(cleaned_data, cfg.to_dict()["data"], mode="after_cleaning", metrics=data_cleaning_metrics)

    else:
        cleaned_data = splitted_data

    one_hot_data = one_hot_encoding(cleaned_data)
    numeric_data = convert_bool_and_cat_to_num(one_hot_data)
    normalized_data = normalization(numeric_data, cfg.data.dataset.sensitive_attributes)

    num_classes = max([len(val[1].squeeze(1).unique()) for val in normalized_data["clients_train"].values()])
    final_data = final_data_preprocessing(normalized_data, cfg, num_classes)

    dummy_data_container = DummyDataContainer(
        final_data["clients_train"],
        final_data["clients_test"],
        final_data["server_test"],
        final_data["clients_train"][0].num_labels,
    )

    data_splitter = DummyDataSplitter(
        dataset=dummy_data_container,
        distribution=cfg.data.distribution.name,
        dist_args=cfg.data.distribution.exclude("name"),
        **cfg.data.others.exclude("client_val_split", "server_test_union", "server_val_split"),
    )

    val_data = {k: v for k, v in final_data.items() if k in ["clients_val", "server_val"]}
    return data_splitter, val_data

def load_data_from_folder(
    data_cfg_dict: dict,
    n_clients: int
):

    load_dir = data_cfg_dict["loading"]["load_dir"]
    assert os.path.exists(load_dir)
    del data_cfg_dict["loading"]

    with open(os.path.join(load_dir, "data_config.yaml"), 'r') as file:
        saved_data_cfg_dict = yaml.safe_load(file)
    del saved_data_cfg_dict["loading"]

    if os.path.exists(os.path.join(load_dir, "data_cleaning_metrics.json")):
        is_data_cleaned = True
    else:
        is_data_cleaned = False
        del data_cfg_dict["cleaning"]
        del saved_data_cfg_dict["cleaning"]

    # Verify if the saved data have the same config
    data_cfg_dict["n_clients"] = n_clients
    assert data_cfg_dict == saved_data_cfg_dict

    data = {
        "clients_train": {},
        "clients_test": {},
        "clients_val": {},
        "server_test": None,
        "server_val": None,
    }

    for id_client in range(n_clients):
        id_client = "client_{}".format(id_client)
        client_path = os.path.join(load_dir, id_client)

        client_X_tr = pd.read_pickle(os.path.join(client_path, "{}_X_tr.pkl".format(id_client)))
        client_y_tr = pd.read_pickle(os.path.join(client_path, "{}_y_tr.pkl".format(id_client)))
        data["clients_train"][id_client] = (client_X_tr, client_y_tr)

        if data_cfg_dict["others"]["client_split"] > 0:
            client_X_te = pd.read_pickle(os.path.join(client_path, "{}_X_te.pkl".format(id_client)))
            client_y_te = pd.read_pickle(os.path.join(client_path, "{}_y_te.pkl".format(id_client)))
            data["clients_test"][id_client] = (client_X_te, client_y_te)
        else:
            data["clients_test"][id_client] = None

        if data_cfg_dict["others"]["client_val_split"] > 0:
            client_X_val = pd.read_pickle(os.path.join(client_path, "{}_X_val.pkl".format(id_client)))
            client_y_val = pd.read_pickle(os.path.join(client_path, "{}_y_val.pkl".format(id_client)))
            data["clients_val"][id_client] = (client_X_val, client_y_val)
        else:
            data["clients_val"][id_client] = None

    if data_cfg_dict["others"]["server_test"] or data_cfg_dict["others"]["server_test_union"]:
        server_path = os.path.join(load_dir, "server")

        server_X_te = pd.read_pickle(os.path.join(server_path, "server_X_te.pkl"))
        server_y_te = pd.read_pickle(os.path.join(server_path, "server_y_te.pkl"))
        data["server_test"] = (server_X_te, server_y_te)

        condition = data_cfg_dict["others"]["server_test_union"] and data_cfg_dict["others"]["client_val_split"] > 0
        if data_cfg_dict["others"]["server_val_split"] > 0 or condition:
            server_X_val = pd.read_pickle(os.path.join(server_path, "server_X_val.pkl"))
            server_y_val = pd.read_pickle(os.path.join(server_path, "server_y_val.pkl"))
            data["server_val"] = (server_X_val, server_y_val)

    return data, is_data_cleaned

def save_data_to_pkl(
    data,
    data_cfg_dict: dict,
    mode: str,
    metrics: dict = {},
):

    save_dir = data_cfg_dict["loading"]["save_dir"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if mode == "before_cleaning":
        subdir = "no_cleaning"
        data_cfg_dict["cleaning"] = None
    else:
        subdir = "{}_cleaning".format(data_cfg_dict["cleaning"]["name"])

    save_subdir = os.path.join(save_dir, subdir)
    if os.path.exists(save_subdir):
        print("Subdir \"{}\" already exists! Data is not saved!".format(save_subdir))
        return 0
    else:
        os.makedirs(save_subdir)

    for id_client, client_tr in data["clients_train"].items():
        client_path = os.path.join(save_subdir, "{}".format(id_client))
        os.mkdir(client_path)

        client_X_tr, client_y_tr = client_tr
        client_X_tr_path = os.path.join(client_path, "{}_X_tr.pkl".format(id_client))
        client_X_tr.to_pickle(client_X_tr_path)
        client_y_tr_path = os.path.join(client_path, "{}_y_tr.pkl".format(id_client))
        client_y_tr.to_pickle(client_y_tr_path)

        client_te = data["clients_test"][id_client]
        if client_te is not None:
            client_X_te, client_y_te = client_te
            client_X_te_path = os.path.join(client_path, "{}_X_te.pkl".format(id_client))
            client_X_te.to_pickle(client_X_te_path)
            client_y_te_path = os.path.join(client_path, "{}_y_te.pkl".format(id_client))
            client_y_te.to_pickle(client_y_te_path)

        client_val = data["clients_val"][id_client]
        if client_val is not None:
            client_X_val, client_y_val = client_val
            client_X_val_path = os.path.join(client_path, "{}_X_val.pkl".format(id_client))
            client_X_val.to_pickle(client_X_val_path)
            client_y_val_path = os.path.join(client_path, "{}_y_val.pkl".format(id_client))
            client_y_val.to_pickle(client_y_val_path)

    server_path = os.path.join(save_subdir, "server")
    os.mkdir(server_path)

    if data["server_test"] is not None:
        server_X_te, server_y_te = data["server_test"]
        server_X_te_path = os.path.join(server_path, "server_X_te.pkl")
        server_X_te.to_pickle(server_X_te_path)
        server_y_te_path = os.path.join(server_path, "server_y_te.pkl")
        server_y_te.to_pickle(server_y_te_path)

    if data["server_val"] is not None:
        server_X_val, server_y_val = data["server_val"]
        server_X_val_path = os.path.join(server_path, "server_X_val.pkl")
        server_X_val.to_pickle(server_X_val_path)
        server_y_val_path = os.path.join(server_path, "server_y_val.pkl")
        server_y_val.to_pickle(server_y_val_path)

    data_cfg_dict["n_clients"] = len(data["clients_train"])

    data_config_path = os.path.join(save_subdir, "data_config.yaml")
    yaml.dump(data_cfg_dict, open(data_config_path, "w"))

    if len(metrics) > 0:
        data_cleaning_metrics_path = os.path.join(save_subdir, "data_cleaning_metrics.json")
        json.dump(metrics, open(data_cleaning_metrics_path, "w"), indent=4)
