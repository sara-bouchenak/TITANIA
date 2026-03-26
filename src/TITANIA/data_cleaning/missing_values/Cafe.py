import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import numpy as np
from copy import deepcopy
import random
import multiprocessing as mp
import itertools

from src.FL_core.data_loading.data_processing import convert_bool_and_cat_to_num

from src.TITANIA.data_cleaning.missing_values.Cafe_utils.fed_imp.sub_modules.server.load_server import load_server
from src.TITANIA.data_cleaning.missing_values.Cafe_utils.fed_imp.sub_modules.client.client_factory import ClientsFactory
from src.TITANIA.data_cleaning.missing_values.Cafe_utils.fed_imp.sub_modules.strategy.strategy_imp import StrategyImputation
from src.TITANIA.data_cleaning.missing_values.Cafe_utils.hyper_params import Hyperparameters

from src.TITANIA.data_cleaning.missing_values.default import MissingValuesDataCleaningMethod


class Cafe(MissingValuesDataCleaningMethod):
    # Cafe does not clean val sets
    ### TODO: calculate n_samples and n_label_errors

    def __init__(self, seed, sensitive_attributes: list[str]):
        super().__init__(sensitive_attributes)
        self.seed = seed
        self.val_data = None

    def clean_errors(self, all_data):
        self.val_data = all_data["server_val"]
        return super().clean_errors(all_data)

    def clean_client_errors(self, clients_data_dict):
        self.n_samples, self.n_detected_errors = self.init_counters()
        cleaned_clients_data = self.clean_missing_values_dict(clients_data_dict)
        cleaning_metrics = self.compute_cleaning_metrics()
        return cleaned_clients_data, cleaning_metrics

    def clean_server_errors(self, server_data):
        # Default missing values cleaning for server data
        return super().clean_server_errors(server_data)

    def clean_missing_values_dict(self, clients_data_dict):
        if clients_data_dict["client_0"] == None:
            return clients_data_dict
        else:
            data_cleaned = self.handle_missing_values_with_cafe(clients_data_dict)
            return data_cleaned

    def handle_missing_values_with_cafe(self, clients_data_dict):

        for id_client, client_data in clients_data_dict.items():
            X, y = client_data
            self.count_n_samples_by_sensitive_attributes(X)
            self.count_n_detected_errors_by_sensitive_attributes(client_data)

        data_copy = clients_data_dict.copy()

        val_data = self.val_data
        val_data_copy=(val_data[0].copy(), val_data[1].copy())

        for key in data_copy.keys():
            data_copy[key] = (data_copy[key][0].reindex(sorted(data_copy[key][0].columns), axis=1),data_copy[key][1])

        ohe_data, ordinal_encoder, categorical_cols = ordinal_encoding({'clients_train':data_copy, "server_val":val_data_copy})
        for key in ohe_data['clients_train'].keys():
            ohe_data['clients_train'][key] = (ohe_data['clients_train'][key][0].reindex(sorted(ohe_data['clients_train'][key][0].columns), axis=1),ohe_data['clients_train'][key][1])
        numeric_data = convert_bool_and_cat_to_num(ohe_data.copy())
        for key in numeric_data['clients_train'].keys():
            numeric_data['clients_train'][key] = (numeric_data['clients_train'][key][0].reindex(sorted(numeric_data['clients_train'][key][0].columns), axis=1),numeric_data['clients_train'][key][1])
        print("original", ohe_data["clients_train"]["client_0"][0].mean())

        normalized_data, scaler = normalization_with_scaler(numeric_data, self.sensitive_attributes, ordinal_encoder.get_feature_names_out())
        for key in normalized_data['clients_train'].keys():
            normalized_data['clients_train'][key] = (normalized_data['clients_train'][key][0].reindex(sorted(normalized_data['clients_train'][key][0].columns), axis=1),normalized_data['clients_train'][key][1])
        encoded_data = normalized_data['clients_train']
        ix = [(row, col) for row in range(encoded_data["client_0"][0].shape[0]) for col in range(encoded_data["client_0"][0].shape[1])]
        encoded_data_list=[]
        for key in encoded_data.keys():
            encoded_data_list.append(pd.concat([encoded_data[key][0],encoded_data[key][1]], axis=1, ignore_index=False))#.to_numpy())
        encoded_data_server=pd.concat([normalized_data['server_val'][0],normalized_data['server_val'][1]], axis=1, ignore_index=False)
        new_X=cafe_main_function(encoded_data_list,encoded_data_server,self.seed)

        cleaned_data=new_X["data"]["imputed_data"]
        cleaned_data_dict={}
        loaded_data=0
        for i in range(len(clients_data_dict)):
            col_names=list(clients_data_dict["client_0"][0].columns.values)
            col_names.sort()
            num_tuples=clients_data_dict["client_"+str(i)][0].shape[0]
            cleaned_data_dict["client_"+str(i)]=(pd.DataFrame(cleaned_data[loaded_data:loaded_data+num_tuples][:,:-1],columns=col_names),pd.DataFrame(cleaned_data[loaded_data:loaded_data+num_tuples][:,-1:]))
            loaded_data+=num_tuples
        list_df = [X for (X, y) in cleaned_data_dict.values()]
        df = pd.concat(list_df, axis=0, ignore_index=True)
        norm_cols = df.columns.tolist()
        norm_cols = [col for col in norm_cols if col not in self.sensitive_attributes]
        norm_cols = [col for col in norm_cols if col not in ordinal_encoder.get_feature_names_out()]
        updated_cleaned_data_dict={}
        for id_client, client_data in cleaned_data_dict.items():
            if client_data != None:
                X, y = client_data
                X[norm_cols] = scaler.inverse_transform(X[norm_cols])
                updated_cleaned_data_dict[id_client] = (X, y)
        for key in updated_cleaned_data_dict.keys():
            updated_cleaned_data_dict[key] = (updated_cleaned_data_dict[key][0].reindex(sorted(updated_cleaned_data_dict[key][0].columns), axis=1),updated_cleaned_data_dict[key][1])
        final_data={}
        for id_client, client_data in updated_cleaned_data_dict.items():
            if client_data != None:
                X, y = client_data
                for col in categorical_cols:
                    numeric_data2=X[col].copy()
                    values=numeric_data2.unique()
                    sorted_list=values.sort()
                one_hot_array = ordinal_encoder.inverse_transform(X[categorical_cols])
                one_hot_df = pd.DataFrame(one_hot_array, columns=ordinal_encoder.get_feature_names_out())
                X_one_hot = pd.concat([X.drop(categorical_cols, axis=1).reset_index(drop=True), one_hot_df], axis=1)
                final_data[id_client] = (X_one_hot, y)
        for key in final_data.keys():
            final_data[key] = (final_data[key][0].reindex(sorted(final_data[key][0].columns), axis=1),final_data[key][1])
        
        return final_data


def ordinal_encoding(data):

    list_df = []
    for key, subdata in data.items():
        if "clients" in key:
            for id_client, client_data in subdata.items():
                if client_data != None:
                    X, y = client_data
                    list_df.append(X)
        else:
            if subdata != None:
                X, y = subdata
                list_df.append(X)
    df = pd.concat(list_df, axis=0, ignore_index=True)
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit(df[categorical_cols])
    for key, subdata in data.items():
        if "clients" in key:
            for id_client, client_data in subdata.items():
                if client_data != None:
                    X, y = client_data
                    one_hot_array = ordinal_encoder.transform(X[categorical_cols])
                    one_hot_df = pd.DataFrame(one_hot_array, columns=ordinal_encoder.get_feature_names_out())
                    X_one_hot = pd.concat([X.drop(categorical_cols, axis=1).reset_index(drop=True), one_hot_df], axis=1)
                    data[key][id_client] = (X_one_hot, y)
        else:
            if subdata != None:
                X, y = subdata
                one_hot_array = ordinal_encoder.transform(X[categorical_cols])
                one_hot_df = pd.DataFrame(one_hot_array, columns=ordinal_encoder.get_feature_names_out())
                X_one_hot = pd.concat([X.drop(categorical_cols, axis=1).reset_index(drop=True), one_hot_df], axis=1)
                data[key] = (X_one_hot, y)

    return data, ordinal_encoder, categorical_cols

def normalization_with_scaler(data, sensitive_attributes, ordinal_columns):

    list_df = [X for (X, y) in data["clients_train"].values()]
    df = pd.concat(list_df, axis=0, ignore_index=True)

    norm_cols = df.columns.tolist()
    norm_cols = [col for col in norm_cols if col not in sensitive_attributes]
    norm_cols = [col for col in norm_cols if col not in ordinal_columns]
    scaler = StandardScaler()
    scaler.fit(df[norm_cols])

    for key, subdata in data.items():
        if "clients" in key:
            for id_client, client_data in subdata.items():
                if client_data != None:
                    X, y = client_data
                    X[norm_cols] = scaler.transform(X[norm_cols])
                    data[key][id_client] = (X, y)
        else:
            if subdata != None:
                X, y = subdata
                X[norm_cols] = scaler.transform(X[norm_cols])
                data[key] = (X, y)

    return data, scaler


def cafe_main_function(train_data, test_data, seed):
    print("changed line 92 of iterative imputation and don't knof if it's right")
    param=None
    num_clients=len(train_data)
    tune_params=False
    mtp=False
    n_rounds=5
    test_size=0.1
    regression=False
    num_processes_configs=5
    random.seed(seed)  # seed for split data
    print('imp_round 20','n_rounds 5')
    configuration={'num_clients': num_clients, 'data': {'dataset_name': 'codon', 'normalize': True}, 'handle_imbalance': None, 'imputation': {'initial_strategy_num': 'mean', 'initial_strategy_cat': 'mode', 'estimator_num': 'ridge_cv', 'estimator_cat': 'logistic_cv', 'imp_evaluation_model': 'logistic', 'imp_evaluation_params': {'tune_params': 'gridsearch'}, 'clip': True}, 'agg_strategy_imp': {'strategy': 'fedavg-s', 'params': {'ms_field': 'missing_cell_pct', 'beta': 0.7}}, 'client_type': 'ice', 'server_type': 'fedavg_pytorch', 'server': {'impute_mode': 'instant', 'imp_round': 20, 'imp_local_epochs': 0, 'pred_round': 0, 'pred_local_epochs': 0, 'model_fit_mode': 'one_shot', 'froze_ms_coefs_round': 100}, 'pred_model': {'model_params': {'model': '2nn', 'num_hiddens': 32, 'model_init_config': None, 'model_other_params': None}, 'train_params': {'batch_size': 128, 'learning_rate': 0.001, 'weight_decay': 0.0001, 'pred_round': 200, 'pred_local_epochs': 3}}, 'experiment': {'n_rounds': 5, 'seed': 102931466, 'mtp': False, 'random_seed': 50, 'num_process': 5, 'test_size': 0.1}, 'track': False, 'tune_params': False, 'prediction': False, 'save_state': True, 'algo_params': {'central': {}, 'central2': {}, 'central_vae': {}, 'central_gain': {}, 'local': {}, 'local_vae': {}, 'local_gain': {}, 'fedavg': {}, 'fedavg_vae': {}, 'fedavg_gain': {}, 'fedmechw': {'client_thres': 1.0, 'alpha': 1.0, 'beta': '0.0,', 'scale_factor': 4}, 'cafe': {'client_thres': 1.0, 'alpha': 0.95, 'gamma': 0.02, 'scale_factor': 4}, 'scale_factor': 6}}
    stat_trackers = []
    if tune_params:
        #param_grid = settings['algo_params_grids'][configuration['agg_strategy_imp']['strategy']]
        print(param_grid)
        keys = param_grid.keys()
        combinations = list(itertools.product(*param_grid.values()))
        params = []
        for comb in combinations:
            params.append(dict(zip(keys, comb)))

        results = []
        # multiprocessing
        num_processes = 5
        chunk_size = len(params) // num_processes
        if chunk_size == 0:
            chunk_size = 1
        # chunks = [exp_configs[i:i + chunk_size] for i in range(0, len(exp_configs), chunk_size)]

        # fed_imp start
        seeds = [seed for i in range(len(params))]
        with mp.Pool(num_processes) as pool:
            process_args = [
                (train_data, test_data, num_clients, i, seed, param)
                for i, param, seed in zip(range(len(params)), params, seeds)
            ]
            process_results = pool.starmap(main_func, process_args, chunksize=chunk_size)

        for ret in process_results:
            results.extend(ret[0])

    elif mtp:
        seed = seed
        seeds = [(seed + 10087 * i) for i in range(n_rounds)]
        #seeds = [i for i in range(n_rounds)]
        rounds = list(range(n_rounds))
        results = []
        # multiprocessing
        num_processes = num_processes_configs
        chunk_size = n_rounds // num_processes
        if chunk_size == 0:
            chunk_size = 1
        # chunks = [exp_configs[i:i + chunk_size] for i in range(0, len(exp_configs), chunk_size)]

        # fed_imp start
        with mp.Pool(num_processes) as pool:
            try:
                process_args = [
                    (train_data, test_data, configuration, num_clients, round, seed)
                    for round, seed in zip(rounds, seeds)]
                process_results = pool.starmap(main_func, process_args, chunksize=chunk_size)
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt, terminating workers")
                pool.terminate()
            else:
                print("Normal termination")
                pool.close()
            
        for ret in process_results:
            results.extend(ret[0])
    else:
        seeds = [(seed + 10087 * i) % (2 ^ 23) for i in range(n_rounds)]
        rounds = list(range(n_rounds))
        results = []
        for round, seed in zip(rounds, seeds):
            rets, stat_tracker = main_func(
                    train_data, test_data, configuration, num_clients, round, seed
            )
            for ret in rets:
                results.append(ret)

    return ret

def main_func(train_data, test_data, configuration, num_clients, round, seed, param=None):
    if param is None:
        repeats = 1
    else:
        repeats = 2
    
    rets, stat_trackers = [], []
    for repeat in range(repeats):
        new_seed = (seed + 10087 * repeat)    
        imp_strategy = configuration['agg_strategy_imp']['strategy']
        strategy_name = imp_strategy.split('-')[0]
        if param:
            params = param
            print(f"Used setted params: {params}")
        else:
            hyper_params = Hyperparameters(
                dataset=configuration['data']['dataset_name'],
                data_partition=[],
                mm_strategy=[],
                num_clients=configuration['num_clients'],
                method=strategy_name
            )
            default_params = hyper_params.get_params()  # get tuned params
            if default_params is None:  # if no tuned params, use default params
                try:
                    params = configuration['algo_params'][strategy_name]
                    print(f"Used default params: {params}")
                except:
                    raise ValueError("No params")
            else:
                params = default_params
                print(f"Used tuned params: {params}")
        print(imp_strategy)
        strategy_imp = StrategyImputation(strategy=imp_strategy, params=params)

        
        # Create Server
        server_type = configuration['server_type']
        server_config = configuration['server']
        server_config["n_cols"] = test_data.shape[1] - 1
    
        pred_config = configuration['pred_model']
        pred_config['model_params']['input_feature_dim'] = test_data.shape[1] - 1
        pred_config['model_params']['output_classes_dim'] = len(np.unique(test_data.iloc[:, -1].values))
        client_factory = ClientsFactory(debug=False)
        clients = client_factory.generate_clients(
            num_clients, train_data, test_data.values, imputation_config=configuration['imputation'], client_type="ice", seed=new_seed
        )
    
        server = load_server(
            server_type,
            clients=clients,
            strategy_imp=strategy_imp,
            server_config=server_config,
            pred_config=pred_config,
            test_data=test_data.values,
            seed=new_seed,
            track=configuration['track'],
            run_prediction=configuration['prediction'],
            persist_data=configuration['save_state'],
        )
    
        # return server
        ret = server.run()
        rets.append(ret)
    
        if configuration['track']:
            stat_trackers.append(deepcopy(server.stats_tracker))
    
        del clients
        del server
        del strategy_imp
    return rets, stat_trackers
