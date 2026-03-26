import random
import math
import numpy as np


def inject_noise(data, cfg, sensitive_attributes):
    for client_name in data["clients_train"].keys():     
        num_tuples=data["clients_train"][client_name][1].shape[0]
        if ("label_errors_percentage" in cfg.keys()) and (cfg.label_errors_percentage > 0):            
            data = inject_label_noise(data, num_tuples, client_name, cfg.label_errors_percentage)
        if ("nan_percentage" in cfg.keys()) and (cfg.nan_percentage > 0):            
            data = inject_nan_noise(data, num_tuples, client_name, cfg.nan_percentage, sensitive_attributes)
    return data

def inject_label_noise(data, num_tuples:int, client_name:str, lbl_error_percentage:float):
    flip_tuples = random.sample(range(1, num_tuples), round(num_tuples*lbl_error_percentage))
    data["clients_train"][client_name][1].loc[flip_tuples] = ~data["clients_train"][client_name][1].loc[flip_tuples]
    return data

def inject_nan_noise(data, num_tuples:int, client_name:str, null_vals_error_percentage:float, sensitive_attributes:list[str]):
    # creates a list of all columns that could have errors
    column_names = list(data["clients_train"][client_name][0].columns.values)
    column_names = [x for x in column_names if x not in sensitive_attributes]
    # creates a list of tuples to flip and then splits them according to the columns
    null_rows = random.sample(range(1, num_tuples), round(num_tuples*null_vals_error_percentage))

    to_remove=[]
    #removes numerical columns, which
    for column_name in column_names:
        if type(data["clients_train"][client_name][0].loc[0, column_name]) is int or type(data["clients_train"][client_name][0].loc[0, column_name]) is np.int64:
            to_remove.append(column_name)

    column_names = [x for x in column_names if x not in to_remove]
    null_row_groups= chunks_fct(null_rows,len(column_names))
    i=0
    # for each column replaces the values
    for column_name in column_names:
        # for the column, find which indexes are already na so that we don't duplicate
        nan_index=data["clients_train"][client_name][0][column_name].isna()
        for removed_index in list(set(list(nan_index.index[nan_index])) & set(null_row_groups[i])):
            while True:
                new_index = math.floor(random.random()*num_tuples)
                #check the new tuple isn't already selected as a row and isn't already a Na
                # could add checks for infinite loops, but not likely to be needed
                if new_index not in null_row_groups and new_index not in list(nan_index.index[nan_index]):
                    break
            null_row_groups[i] = [new_index if x==removed_index else x for x in null_row_groups[i]]

        data["clients_train"][client_name][0].loc[null_row_groups[i], column_name] = np.nan

        i+=1 # different from the column's number because we exclude sensitive attributes

    return data

def chunks_fct(list_coll, n):
    #creates chunks, all equal size except for the last which will be smaller, each a list of lists
    new_list = [0]*n
    list_len = len(list_coll)
    for i in range(n):
        new_list[i] = list_coll[(i*math.ceil(list_len/n)):(min((i+1)*math.ceil(list_len/n),list_len-1))]
    return new_list
