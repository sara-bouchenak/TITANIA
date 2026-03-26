import pandas as pd
import numpy as np
import copy
import random
import torch
from torch.utils.data import Subset
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from argparse import Namespace

from src.TITANIA.data_cleaning.label_errors.default import LabelErrorsDataCleaningMethod
from src.TITANIA.data_cleaning.label_errors.FedCorr_utils.local_training import LocalUpdate
from src.TITANIA.data_cleaning.label_errors.FedCorr_utils.fedavg import FedAvg
from src.TITANIA.data_cleaning.label_errors.FedCorr_utils.util import lid_term, get_output
from src.TITANIA.data_cleaning.label_errors.FedCorr_utils.torchvision_tab_datasets import TabularDataset

from src.FL_core.utils.net import MLP

from src.FL_core.data_loading.data_processing import one_hot_encoding, normalization, convert_bool_and_cat_to_num


class FedCorr(LabelErrorsDataCleaningMethod):

    ### TODO: calculate n_samples and n_label_errors

    def __init__(
            self,
            sensitive_attributes: list[str],
            correction_mode: str,
            seed,
    ):
        super().__init__(sensitive_attributes)
        self.label_errors_correction_mode = correction_mode
        self.seed = seed

    def clean_client_errors(self, clients_data_dict):
        self.n_samples, self.n_label_errors = self.init_counters()
        cleaned_clients_data = self.clean_label_errors_dict(clients_data_dict)
        cleaning_metrics = self.compute_cleaning_metrics()
        return cleaned_clients_data, cleaning_metrics

    def clean_server_errors(self, server_data):
        # No cleaning label errors for server data with FedCorr
        cleaning_metrics = {}
        return server_data, cleaning_metrics

    def clean_label_errors_dict(self, clients_data_dict):
        if clients_data_dict["client_0"] == None:
            return clients_data_dict
        else:
            data_cleaned = self.handle_label_errors_with_fedcorr(clients_data_dict)
            return data_cleaned

    def handle_label_errors_with_fedcorr(self, clients_data_dict):

        for id_client, client_data in clients_data_dict.items():
            X, y = client_data
            self.count_n_samples_by_sensitive_attributes(X)

        data_copy = clients_data_dict.copy()
        ohe_data = one_hot_encoding({'clients_train':data_copy})
        numeric_data = convert_bool_and_cat_to_num(ohe_data)
        normalized_data = normalization(numeric_data, self.sensitive_attributes)
        encoded_data = normalized_data['clients_train']

        X_global_train=[]
        y_global_train=[]
    
        user_groups=[0]*len(encoded_data)
        num_elements=0
        for user_id, (id_client, client_data) in enumerate(encoded_data.items()):
            X, y = client_data
            X_global_train.append(X.copy())
            y_global_train.append(y.copy())
            user_groups[user_id] = set()
            for j in range(X.shape[0]):
                user_groups[user_id].add(j+num_elements)
            num_elements+=X.shape[0]
        X_global_train = pd.concat(X_global_train, axis=0, ignore_index=True)
        y_global_train = pd.concat(y_global_train, axis=0, ignore_index=True)
    
        y_relabelled = fedCorr_main_function(X_global_train, y_global_train, user_groups, self.seed)

        mode = self.label_errors_correction_mode
        new_data = clients_data_dict.copy()
        num_elements=0
        for id_client, client_tr_data in clients_data_dict.items():
            X, y = client_tr_data
            if mode == "remove":
                delete=[]
                for i in range(len(y)):

                    if y[y.columns.values[0]][i]!=y_relabelled[num_elements+i]:
                        delete.append(i)
                new_data[id_client] = (X.drop(delete), y.drop(delete))
                num_elements += X.shape[0]
                #print(len(X),len(X.drop(delete)))

            else:
                new_y = pd.DataFrame({"income": y_relabelled[num_elements:(X.shape[0]+num_elements)]})
                new_data[id_client] = (X, new_y)
                num_elements += X.shape[0]

        return new_data


def fedCorr_main_function(X_global_train, y_global_train, user_groups, seed) -> pd.DataFrame:

    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print("We have ",len(user_groups)," clients")
    args = Namespace(iteration1=5, rounds1=50, rounds2=50, local_ep=1, frac1=max(0.1,1/len(user_groups)), frac2=0.1, num_users=len(user_groups), local_bs=10, lr=0.03, momentum=0.5, beta=0, LID_k=20, level_n_system=0.4, level_n_lowerb=0.5, relabel_ratio=0.5, confidence_thres=0.5, clean_set_thres=0.1, fine_tuning=True, correction=True, pretrained=False, iid=False, non_iid_prob_class=0.7, alpha_dirichlet=10, num_classes=2, seed=seed, mixup=False, alpha=1)
    args.device = torch.device('cpu')

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    dataset_train = TabularDataset(X_global_train, y_global_train)
    #dataset_test = TabularDataset(X_global_test, y_global_test)
    dict_users = user_groups
    # ---------------------Add Noise ---------------------------
    y_train = np.array(dataset_train.targets)
    dataset_train.targets = y_train

    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)

    # build model
    netglob = MLP(input_size=len(np.array(dataset_train.data)[0]), num_classes= 2, n_hidden_layer_1= 64, n_hidden_layer_2= 32)
    net_local = MLP(input_size=len(np.array(dataset_train.data)[0]), num_classes= 2, n_hidden_layer_1= 64, n_hidden_layer_2= 32)
    criterion = nn.CrossEntropyLoss(reduction='none')
    LID_accumulative_client = np.zeros(args.num_users)

    for iteration in range(args.iteration1):
        print("iteration",iteration)
        LID_whole = np.zeros(len(y_train))
        loss_whole = np.zeros(len(y_train))
        LID_client = np.zeros(args.num_users)
        loss_accumulative_whole = np.zeros(len(y_train))

        # ---------Broadcast global model----------------------
        if iteration == 0:
            mu_list = np.zeros(args.num_users)
        else:
            mu_list = estimated_noisy_level

        prob = [1 / args.num_users] * args.num_users

        for _ in range(int(1/args.frac1)):
            idxs_users = np.random.choice(range(args.num_users), int(args.num_users*args.frac1), p=prob)
            w_locals = []
            for idx in idxs_users:
                prob[idx] = 0
                if sum(prob) > 0:
                    prob = [prob[i] / sum(prob) for i in range(len(prob))]

                net_local.load_state_dict(netglob.state_dict())
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)

                # proximal term operation
                mu_i = mu_list[idx]
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=sample_idx)
                w, loss = local.update_weights(net=copy.deepcopy(net_local).to(args.device), seed=seed,
                                                w_g=netglob.to(args.device), epoch=args.local_ep, mu=mu_i)
                net_local.load_state_dict(copy.deepcopy(w))
                w_locals.append(copy.deepcopy(w))
                #acc_t = globaltest(copy.deepcopy(net_local).to(args.device), dataset_test, args)    
                local_output, loss = get_output(loader, net_local.to(args.device), args, False, criterion)
                LID_local = list(lid_term(local_output, local_output))
                LID_whole[sample_idx] = LID_local
                loss_whole[sample_idx] = loss
                LID_client[idx] = np.mean(LID_local)

            dict_len = [len(dict_users[idx]) for idx in idxs_users]
            w_glob = FedAvg(w_locals, dict_len)

            netglob.load_state_dict(copy.deepcopy(w_glob))

        LID_accumulative_client = LID_accumulative_client + np.array(LID_client)
        loss_accumulative_whole = loss_accumulative_whole + np.array(loss_whole)

        # Apply Gaussian Mixture Model to LID
        try:
            gmm_LID_accumulative = GaussianMixture(n_components=2, random_state=seed).fit(
                np.array(LID_accumulative_client).reshape(-1, 1))
        except Exception as e:
            print(LID_accumulative_client)
            print(np.array(LID_accumulative_client).reshape(-1, 1))
            print(e)
            raise ValueError()
        labels_LID_accumulative = gmm_LID_accumulative.predict(np.array(LID_accumulative_client).reshape(-1, 1))
        clean_label = np.argsort(gmm_LID_accumulative.means_[:, 0])[0]

        noisy_set = np.where(labels_LID_accumulative != clean_label)[0]
        clean_set = np.where(labels_LID_accumulative == clean_label)[0]

        estimated_noisy_level = np.zeros(args.num_users)

        for client_id in noisy_set:
            sample_idx = np.array(list(dict_users[client_id]))

            loss = np.array(loss_accumulative_whole[sample_idx])
            gmm_loss = GaussianMixture(n_components=2, random_state=seed).fit(np.array(loss).reshape(-1, 1))
            labels_loss = gmm_loss.predict(np.array(loss).reshape(-1, 1))
            gmm_clean_label_loss = np.argsort(gmm_loss.means_[:, 0])[0]

            pred_n = np.where(labels_loss.flatten() != gmm_clean_label_loss)[0]
            estimated_noisy_level[client_id] = len(pred_n) / len(sample_idx)
            y_train_new = np.array(dataset_train.targets)

        if args.correction:
            for idx in noisy_set:
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
                loss = np.array(loss_accumulative_whole[sample_idx])
                local_output, _ = get_output(loader, netglob.to(args.device), args, False, criterion)
                relabel_idx = (-loss).argsort()[:int(len(sample_idx) * estimated_noisy_level[idx] * args.relabel_ratio)]
                relabel_idx = list(set(np.where(np.max(local_output, axis=1) > args.confidence_thres)[0]) & set(relabel_idx))


                y_train_new = np.array(dataset_train.targets)
                y_train_new[sample_idx[relabel_idx]] = np.argmax(local_output, axis=1)[relabel_idx]
                dataset_train.targets = y_train_new

    # reset the beta,
    args.beta = 0

    # ---------------------------- second stage training -------------------------------
    if args.fine_tuning:
        selected_clean_idx = np.where(estimated_noisy_level <= args.clean_set_thres)[0]
    
        prob = np.zeros(args.num_users) # np.zeros(100)
        prob[selected_clean_idx] = 1 / len(selected_clean_idx)
        m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
        m = min(m, len(selected_clean_idx))
        netglob = copy.deepcopy(netglob)
        # add fl training
        for rnd in range(args.rounds1):
            w_locals, loss_locals = [], []
            idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
            for idx in idxs_users:  # training over the subset
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), seed=seed,
                                                            w_g=netglob.to(args.device), epoch=args.local_ep,  mu=0)
                w_locals.append(copy.deepcopy(w_local))  # store every updated model
                loss_locals.append(copy.deepcopy(loss_local))

            dict_len = [len(dict_users[idx]) for idx in idxs_users]
            w_glob_fl = FedAvg(w_locals, dict_len)
            netglob.load_state_dict(copy.deepcopy(w_glob_fl))
    
            #acc_s2 = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)

        if args.correction:
            relabel_idx_whole = []
            for idx in noisy_set:
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
                glob_output, _ = get_output(loader, netglob.to(args.device), args, False, criterion)
                y_predicted = np.argmax(glob_output, axis=1)
                relabel_idx = np.where(np.max(glob_output, axis=1) > args.confidence_thres)[0]
                y_train_new = np.array(dataset_train.targets)
                y_train_new[sample_idx[relabel_idx]] = y_predicted[relabel_idx]
                dataset_train.targets = y_train_new
    same=0
    dif=0
    for i in range(len(y_train)):
        if y_train[i]==y_train_new[i]:
            same+=1
        else:
            dif+=1
    print(same,dif)
    return y_train_new
