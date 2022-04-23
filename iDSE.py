import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
import copy
import os
import random
import time
import math
from datetime import datetime
from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd
import seaborn as sns
import heapq
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit

# 1 for 5%, 2 for 10%
sample_id = 20
sample_id_max = sample_id

gen_choose_list_flag = 0
choose_from_dsp = 1

sample_num = 1
CASE_MAX_NUM = 52
#416.1/2/3 is ignored

#372-exe2
#373-pd
#374-pb
#375-pi
#376-pi-pb
#379-exe2-pd-pi-pb
specific_version = ['9-37', '9-372', '9-373', '9-376', '9-379']
bench_info_cpi = np.zeros((1+CASE_MAX_NUM,30,len(specific_version)), dtype=float)
bench_info_power = np.zeros((1+CASE_MAX_NUM,30,len(specific_version)), dtype=float)
bench_info = np.zeros((1+CASE_MAX_NUM,30), dtype=dict)

BENCH_ID_INDEX = 0
BENCH_SIMPOINT_INDEX = BENCH_ID_INDEX + 1
CASE_VERSION_INDEX = BENCH_SIMPOINT_INDEX + 1

#CPI_mode = 1
simpoint_mode = 1
area_mode = 0
input_enable = 0
inst_radio_mode = 1
test_fig_mode = 0

#bench_indivisual_model = None
#bench_indivisual_model = '400.1'
#bench_indivisual_model = '483.1'

bench_indivisual_model = '403.2'
#bench_indivisual_model = '473.1'
simpoint_id = 1

#plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.size'] = '10'
plt.rcParams['font.weight'] = 'bold'

log_name = ''
if bench_indivisual_model:
    log_name += bench_indivisual_model + '-' + str(1)
log_name += '-' + str(sample_id) + '-' + str(area_mode)
log_name += '-train_eval.log'
log_file = open(log_name, 'w')

input_length_max = 0

bench_array = [
    ('xxx', 'xxx', '0'),
    ('ref', '400.1', '1'),
    ('ref', '400.2', '2'),
    ('ref', '400.3', '3'),
    ('ref', '401.1', '4'),
    ('ref', '401.2', '5'),
    ('ref', '401.3', '6'),
    ('ref', '401.4', '7'),
    ('ref', '401.5', '8'),
    ('ref', '401.6', '9'),
    ('ref', '403.1', '10'),
    ('ref', '403.2', '11'),
    ('ref', '403.3', '12'),
    ('ref', '403.4', '13'),
    ('ref', '403.5', '14'),
    ('ref', '403.6', '15'),
    ('ref', '403.7', '16'),
    ('ref', '403.8', '17'),
    ('ref', '403.9', '18'),
    ('ref', '429.1', '19'),
    ('ref', '445.1', '20'),
    ('ref', '445.2', '21'),
    ('ref', '445.3', '22'),
    ('ref', '445.4', '23'),
    ('ref', '445.5', '24'),
    ('ref', '456.1', '25'),
    ('ref', '456.2', '26'),
    ('ref', '458.1', '27'),
    ('ref', '462.1', '28'),
    ('ref', '464.1', '29'),
    ('ref', '464.2', '30'),
    ('ref', '464.3', '31'),
    ('ref', '471.1', '32'),
    ('ref', '473.1', '33'),
    ('ref', '473.2', '34'),
    ('ref', '483.1', '35'),
    ('ref', '433.1', '36'),
    ('ref', '444.1', '37'),
    ('ref', '470.1', '38'),
    ('ref', '410.1', '39'),
    ('ref', '434.1', '40'),
    ('ref', '435.1', '41'),
    ('ref', '436.1', '42'),
    ('ref', '437.1', '43'),
    ('ref', '447.1', '44'),
    ('ref', '450.1', '45'),
    ('ref', '450.2', '46'),
    ('ref', '453.1', '47'),
    ('ref', '454.1', '48'),
    ('ref', '459.1', '49'),
    ('ref', '465.1', '50'),
    ('ref', '482.1', '51'),
    ('ref', '481.1', '52'),
    ('ref', '416.1', '53'),
    ('ref', '416.2', '54'),
    ('ref', '416.3', '55')]

sub_bench_array = [
    #int
    '400.1',
    '462.1',
    '473.2',
    '483.1',
    #fp
    '437.1',
    '454.1',
    '436.1',
    '447.1',
    '470.1',
    '453.1',
]

SIMULATOR_CYCLES_PER_SECOND_map = {
'0' : 1000000000,
'1' : 1500000000,
'2' : 2000000000, #2GHz
'3' : 3000000000,
}

IFQ_SIZE_map = {
'0' : 8,
'1' : 16,
}

DECODEQ_SIZE_map = {
'0' : 8,
'1' : 16,
}
FETCH_WIDTH_map = {
'0' : 2,
'1' : 4,
'2' : 8,
'3' : 16,
}
DECODE_WIDTH_map = {
'0' : 2,
'1' : 3,
'2' : 4,
'3' : 5,
}

DISPATCH_WIDTH_map = {
'0' : 2,
'1' : 4,
'2' : 5,
'3' : 6,
}

COMMIT_WIDTH_map = {
'0' : 2,
'1' : 4,
'2' : 6,
'3' : 8,
}

PHY_GPR_NUM_map = {
'0' : 40,
'1' : 64,
'2' : 128,
'3' : 180,
}

PHY_FGPR_NUM_map = {
'0' : 40,
'1' : 64,
'2' : 128,
'3' : 180,
}

GPR_WRITEBACK_WIDTH_map = {
'0' : 2,
'1' : 4,
}

FGPR_WRITEBACK_WIDTH_map = {
'0' : 2,
'1' : 4,
}

RUU_SIZE_MAX_map = {
'0' : 32,
'1' : 64,
'2' : 128,
'3' : 256,
}

INT_BP_map = {
'0': 1,
'1': 2,
}

INT_ALU_map = {
'0': 1,
'1': 2,
}

INT_MULT_map = {
'0': 1,
'1': 2,
}

INT_MULT_OP_LAT_map = {
'0' : 2,
'1' : 4
}

INT_MULT_ISSUE_LAT_map = {
'0' : 4,
'1' : 1,
}

INT_DIV_OP_LAT_map = {
'0' : 8,
'1' : 16,
}

INT_DIV_ISSUE_LAT_map = {
'0' : 16,
'1' : 1,
}

FP_ALU_map = {
'0' : 1,
'1' : 2,
}

FP_ALU_MULT_map = {
'0' : 1,
'1' : 2,
}

FP_MULT_DIV_map = {
'0' : 1,
'1' : 2,
}

FP_ALU_MULT_DIV_map = {
'0' : 0,
'1' : 1,
}

FP_MULT_OP_LAT_map = {
'0' : 2,
'1' : 4,
}

FP_MULT_ISSUE_LAT_map = {
'0' : 4,
'1' : 1,
}

FP_DIV_OP_LAT_map = {
'0' : 8,
'1' : 16,
}

FP_DIV_ISSUE_LAT_map = {
'0' : 16,
'1' : 1,
}
'''
FP_SQRT_OP_LAT_map = {
'0' : 4,
'1' : 1,
}

FP_SQRT_ISSUE_LAT_map = {
'0' : 4,
'1' : 1,
}
'''

LOAD_PORT_WIDTH_map = {
'0': 1,
'1': 2,
}

STORE_PORT_WIDTH_map = {
'0': 1,
'1': 2,
}

LOAD_STORE_PORT_WIDTH_map = {
'0': 0,
'1': 2,
}

LOAD_QUEUE_SIZE_map = {
'0' : 10,
'1' : 30,
'2' : 60,
'3' : 90,
}

STORE_QUEUE_SIZE_map = {
'0' : 10,
'1' : 30,
'2' : 60,
'3' : 90,
}

BPRED_map = {
'0' : 'tiDSE',
'1' : 'tage'
}

RAS_SIZE_map = {
'0' : 8,
'1' : 16,
}

L1_ICACHE_SET_map = {
'0' : 64,
'1' : 128,
'2' : 256,
}

L1_ICACHE_ASSOC_map = {
'0' : 2,
'1' : 4,
'2' : 8,
}

L1_DCACHE_SET_map = {
'0' : 64,
'1' : 128,
'2' : 256,
}

L1_DCACHE_ASSOC_map = {
'0' : 2,
'1' : 4,
'2' : 8,
}

L1_DCACHE_WRITEBACK_map = {
'0': 0,
'1': 1
}

L2_CACHE_SET_map = {
'0' : 128,
'1' : 1024,
}

L2_CACHE_ASSOC_map = {
'0' : 4,
'1' : 8,
}

LLC_map = {
'0' : 2,
#'1' : 3,
}

version_map_id = {
'IFQ_SIZE': [0, len(IFQ_SIZE_map)],
'DECODEQ_SIZE' : [1, len(DECODEQ_SIZE_map)],
'FETCH_WIDTH' : [2, len(FETCH_WIDTH_map)],
'DECODE_WIDTH' : [3, len(DECODE_WIDTH_map)],
'DISPATCH_WIDTH' : [4, len(DISPATCH_WIDTH_map)],
'COMMIT_WIDTH' : [5, len(COMMIT_WIDTH_map)],
'PHY_GPR_NUM': [6, len(PHY_GPR_NUM_map)],
'PHY_FGPR_NUM': [7, len(PHY_FGPR_NUM_map)],
'GPR_WRITEBACK_WIDTH': [8, len(GPR_WRITEBACK_WIDTH_map)],
'FGPR_WRITEBACK_WIDTH': [9, len(FGPR_WRITEBACK_WIDTH_map)],
'RUU_SIZE_MAX': [10, len(RUU_SIZE_MAX_map)],
'INT_BP' : [11, len(INT_BP_map)],
'INT_ALU': [12, len(INT_ALU_map)],
'INT_MULT' : [13, len(INT_MULT_map)],
'INT_MULT_OP_LAT' : [14, len(INT_MULT_OP_LAT_map)],
'INT_MULT_ISSUE_LAT' : [15, len(INT_MULT_ISSUE_LAT_map)],
'INT_DIV_OP_LAT': [16, len(INT_DIV_OP_LAT_map)],
'INT_DIV_ISSUE_LAT': [17, len(INT_DIV_ISSUE_LAT_map)],
'FP_ALU': [18, len(FP_ALU_map)],
'FP_ALU_MULT': [19, len(FP_ALU_MULT_map)],
'FP_MULT_DIV': [20, len(FP_MULT_DIV_map)],
'FP_ALU_MULT_DIV': [21, len(FP_ALU_MULT_DIV_map)],
'FP_MULT_OP_LAT': [22, len(FP_MULT_OP_LAT_map)],
'FP_MULT_ISSUE_LAT': [23, len(FP_MULT_ISSUE_LAT_map)],
'FP_DIV_OP_LAT': [24, len(FP_DIV_OP_LAT_map)],
'FP_DIV_ISSUE_LAT': [25, len(FP_DIV_ISSUE_LAT_map)],
#'FP_SQRT_OP_LAT': 25,
#'FP_SQRT_ISSUE_LAT': 26,
'LOAD_PORT_WIDTH' : [26, len(LOAD_PORT_WIDTH_map)],
'STORE_PORT_WIDTH': [27, len(STORE_PORT_WIDTH_map)],
'LOAD_STORE_PORT_WIDTH': [28, len(LOAD_STORE_PORT_WIDTH_map)],
'LOAD_QUEUE_SIZE': [29, len(LOAD_QUEUE_SIZE_map)],
'STORE_QUEUE_SIZE': [30, len(STORE_QUEUE_SIZE_map)],
'BPRED': [31, len(BPRED_map)],
'RAS_SIZE' : [32, len(RAS_SIZE_map)],
'L1_ICACHE_SET' : [33, len(L1_ICACHE_SET_map)],
'L1_ICACHE_ASSOC' : [34, len(L1_ICACHE_ASSOC_map)],
'L1_DCACHE_SET' : [35, len(L1_DCACHE_SET_map)],
'L1_DCACHE_ASSOC' : [36, len(L1_DCACHE_ASSOC_map)],
'L1_DCACHE_WRITEBACK' : [37, len(L1_DCACHE_WRITEBACK_map)],
'L2_CACHE_SET' : [38, len(L2_CACHE_SET_map)],
'L2_CACHE_ASSOC': [39, len(L2_CACHE_ASSOC_map)],
'LLC': [40, len(LLC_map)],
'max' : 41,
}

DEF_FREQ = 0
DEF_IFQ = 1
DEF_DECODEQ = 2
DEF_FETCH_WIDTH = 3
DEF_DECODE_WIDTH = 4
DEF_DISPATCH_WIDTH = 5
DEF_COMMIT_WIDHT = 6
DEF_GPR = 7
DEF_FGPR = 8
DEF_GPR_WRITEBACK = 9
DEF_FGPR_WRITEBACK = 10
DEF_RUU_SIZE_MAX = 11
DEF_INT_BP = 12
DEF_FP_ALU = 19
DEF_LOAD_PORT_WIDTH = 29

DEF_MULTI_BTB = 35
DEF_BPRED = 36
DEF_L0_ICACHE = 38
DEF_EXECUTE_RECOVER = 39
DEF_RAW_LOAD_PRED = 40
DEF_PREFETCH_INST = 41
DEF_PREFETCH_DATA = 42
DEF_L1_ICACHE_SET = 43
DEF_L1_DCACHE_SET = 45
DEF_L2_CACHE_SET = 48
#49

#input_mask_array = [DEF_MULTI_BTB, DEF_L0_ICACHE, DEF_EXECUTE_RECOVER, DEF_RAW_LOAD_PRED, DEF_PREFETCH_INST, DEF_PREFETCH_DATA]
input_mask_array = [
DEF_IFQ
, DEF_DECODEQ
, DEF_FETCH_WIDTH
, DEF_DECODE_WIDTH
, DEF_MULTI_BTB
, DEF_L0_ICACHE
, DEF_EXECUTE_RECOVER
, DEF_RAW_LOAD_PRED
, DEF_PREFETCH_INST
, DEF_PREFETCH_DATA]

input_enable_array = [
DEF_DISPATCH_WIDTH
, DEF_INT_BP
, DEF_FP_ALU
, DEF_LOAD_PORT_WIDTH
, DEF_BPRED
, DEF_L1_ICACHE_SET
, DEF_L1_DCACHE_SET
, DEF_L2_CACHE_SET
]


def read_area(configs):
    area_file = open('area_core.txt', "r")
    for area_str in area_file:
        area_str_array = area_str.split()
        version = area_str_array[0]
        area = area_str_array[1]
        for config in configs:
            if version == config['name']:
                config['area'] = float(area)
    area_file.close()

def read_config(config_dir):
    configs = []
    #config_list = [config for config in ]

    #config_dir_name = config_dir
    for config_dir_name in os.listdir(config_dir):
        #if config_dir_name[0:5] == '11333':
        #    continue
        config_file = open(config_dir + config_dir_name, "r", encoding='utf-8')
        try:
            param_list = [float(i) for i in config_file.read().strip().split(" ")]
        except:
            print(config_dir_name + ' failed')
        global input_length_max
        if 0 == input_length_max:
            input_length_max = len(param_list)
        elif len(param_list) != input_length_max:
            print(config_dir_name + ' len != ' + str(input_length_max))
            #exit(1)
        #print(param_list)
        configs += [{'name': config_dir_name.split('.')[0], 'params': param_list, 'area': 0.0}]
        config_file.close()
    return configs

configs_all = read_config('./config_history_all/')
if area_mode:
    read_area(configs_all)

def get_config(configs, config_name):
    for config in configs:
        if config_name == config['name']:
            return config
    #print('no fould config ' + config_name)
    return None

def get_runned_config(bench_id, runned_config_list_path):
    bench_name = bench_array[bench_id][1].split('.')
    runned_config_list_random = []
    for runned_config_file in os.listdir(runned_config_list_path):
        if bench_name[0] in runned_config_file and ('_' + bench_name[1] + '_' + bench_array[bench_id][0]) in runned_config_file:
            runned_config_file = open(runned_config_list_path + '/' + runned_config_file, "r", encoding='utf-8')
            for case in runned_config_file:
                runned_config_list_random.append(case.strip('\n'))
            runned_config_file.close()
            return runned_config_list_random

def choose_runned_case2(choose_limit, all_input, cpi_label, power_label, y_label, real_pareto_points_config):
    choose_case = []
    choose_cpi_label = []
    choose_power_label = []
    choose_y_label = []
    old_bench_id = 0
    bench_name = bench_indivisual_model
    global simpoint_id
    bench_file = bench_indivisual_model + '-ref-' + str(simpoint_id)
    [bench_id, simpoint_id] = get_bench_id(bench_file)
    runned_config_list_choose = gen_choose_list(bench_name, bench_id, simpoint_id, y_label, real_pareto_points_config)
    #choose_limit = int(len(y_label) * 0.9)
    choose_one_in_pareto = 0
    for choose_one in runned_config_list_choose:
        found = 0
        for case, cpi, power, case_y in zip(all_input, cpi_label, power_label, y_label):
            if case_y[BENCH_ID_INDEX] == bench_id and case_y[BENCH_SIMPOINT_INDEX] == simpoint_id:
                if case_y[CASE_VERSION_INDEX] == choose_one:
                    choose_case.append(case)
                    choose_cpi_label.append(cpi)
                    choose_power_label.append(power)
                    choose_y_label.append(case_y)
                    choose_limit -= 1
                    found = 1
                    if choose_one in real_pareto_points_config:
                        choose_one_in_pareto += 1
                        print('choose_one_in_pareto ++ = ' + str(choose_one_in_pareto))
            if found:
                break
        if 0 == found:
            print('version not found in choose ', choose_one)
        if 0 == choose_limit:
            if 1 < sample_id:
                print('warn should not be choose_limit == 0')
            break

    print('choose_one_in_pareto = ' + str(choose_one_in_pareto) + ' / ' + str(len(runned_config_list_choose)))
    if area_mode:
        hit_file = open('log/' + 'area-' + 'hit_file_choose.txt', 'a')
    else:
        hit_file = open('log/' + 'power-' + 'hit_file_choose.txt', 'a')
    hit_file.write('choose_one_in_pareto = ' + str(choose_one_in_pareto) + ' / ' + str(len(choose_case)) + ' / ' + str(len(real_pareto_points_config)) + '\n')
    hit_file.close()
    return [choose_case, choose_cpi_label, choose_power_label, choose_y_label] 

def get_bench_id(bench_file):
    bench_name_str = bench_file.split('-')
    bench_name = bench_name_str[0]
    simpoint_id = 0
    bench_id = 0
    if (2 < len(bench_name_str)):
        simpoint_id = int(bench_name_str[2].split('.')[0])
    for bench_name_cmp in bench_array:
        if bench_name_cmp[1] in bench_name:
            break
        bench_id += 1
    if len(bench_array) < bench_id:
        print(bench_name +' not found')
    return [bench_id, simpoint_id]

def gen_choose_list(bench_name, bench_id, simpoint_id, y_label,real_pareto_points_config):
    choose_list = []
    version_gen_num = 0
    filename = 'choose/' + bench_name + '_ref_simpointid-' + str(simpoint_id)
    version_gen_in_pareto = 0
    if gen_choose_list_flag:
        choose_list_file = open(filename + '.txt', 'w')
        while version_gen_num < 2000:
            #if 0 == version_gen_num:
            #    version_gen = gen_version_choose(DISPATCH_WIDTH_index = 0, exe_int = 0, exe_fp = 0, lsq = 0, dcache = 0, icache = 0, bp = 0, l2cache = 0)
            #elif 1 == version_gen_num:
            #    version_gen = gen_version_choose(DISPATCH_WIDTH_index = 2, exe_int = 1, exe_fp = 1, lsq = 3, dcache = 2, icache = 2, bp = 1, l2cache = 1)
            #else:
            version_gen = gen_choose(bench_id, simpoint_id)
            version_check_exit = 0
            if version_gen in choose_list:
                version_check_exit += 1
            valid_config_flag = False
            for y_label_check in y_label:
                if version_gen == y_label_check[CASE_VERSION_INDEX]:
                    valid_config_flag = True
                    break
            #valid_config_flag                          
            if 0 == version_check_exit and valid_config_flag:
                choose_list.append(version_gen)
                choose_list_file.write(version_gen + '\n')
                choose_list_file.flush()
                version_gen_num += 1
                if version_gen in real_pareto_points_config:
                    version_gen_in_pareto += 1
                    print('version_gen_in_pareto ++ = ' + str(version_gen_in_pareto) + '/' + str(version_gen_num))
        choose_list_file.close()
    else:
        if choose_from_dsp and (1 < sample_id):
            for sample_id_iter in range(0, sample_id):
                filename_dsp = get_new_add_dsp_filename(opt = '_opt', simpoint_id = simpoint_id, sample_id = sample_id_iter)
                choose_list_file = open(filename_dsp, 'r')
                head_flag = True
                for choosen_version_str in choose_list_file:
                    if 0 == sample_id_iter:
                        versino_add = choosen_version_str.strip('\n')
                    else:
                        choosen_version = choosen_version_str.split(' ')
                        if head_flag:
                            head_flag = False
                            continue
                        versino_add = choosen_version[2].strip('\n')
                    choose_list.append(versino_add)
                    #version_gen_num += 1
                choose_list_file.close()
                print('sample_id ', sample_id,' iter ' , sample_id_iter, 'add choose_list to # ', len(choose_list))
        else:
            choose_list_file = open(filename + '.txt', 'r')
            for choosen_version in choose_list_file:
                choose_list.append(choosen_version.strip('\n'))
                #version_gen_num += 1
            choose_list_file.close()
    return choose_list

def get_dsp_version_to_next(opt, simpoint_id):
    filename = get_new_add_dsp_filename(opt, simpoint_id)
    version_list = []
    if os.path.exists(filename):
        file = open(filename, 'r')
        for version_iter_str in file:
            version_iter = version_iter_str.split(' ')
            version_list.append(version_iter[2].strip('\n'))
        file.close()
    return version_list

def get_dsp_point(opt, simpoint_id):
    filename = get_dsp_filename(opt, simpoint_id, sample_id)
    x = []
    y = []
    version_list = []
    if os.path.exists(filename):
        file = open(filename, 'r')
        for version_iter_str in file:
            version_iter = version_iter_str.split(' ')
            y.append(float(version_iter[0]))
            x.append(float(version_iter[1]))
            version_list.append(version_iter[2].strip('\n'))
        file.close()
    else:
        print(filename + 'not found')
    return [x, y, version_list]
    

def dsp_data_filer(dps_version_list, all_input, final_cpi_labels, final_power_labels, real_y_label_array):
    train_data_opt_dsp = []
    train_cpi_opt_dsp = []
    train_power_opt_dsp = []
    train_y_label_opt_dsp = []
    for case,cpi,power,y_label in zip(all_input, final_cpi_labels, final_power_labels, real_y_label_array):
        if y_label[CASE_VERSION_INDEX] in dps_version_list:
            train_data_opt_dsp.append(case)
            train_cpi_opt_dsp.append(cpi)
            train_power_opt_dsp.append(power)
            train_y_label_opt_dsp.append(y_label)      
    return [train_data_opt_dsp, train_cpi_opt_dsp, train_power_opt_dsp, train_y_label_opt_dsp]

  #if (0.001 < fp_ratio) and (random.random() < (bench_info_transform[1])):

def gen_choose(bench_id, simpoint_id):
  DISPATCH_WIDTH_index = 0 + int(random.random() * 4)
  #bench_info = get_bench_info_vector(program, case)
  if 1:
      bench_info_ratios = bench_info_cpi[bench_id][simpoint_id]
      for version_iter in bench_info_cpi[bench_id][simpoint_id]:
        if version_iter <= 0:
            print(version_iter + ' is 0 ,error')
  else:
      bench_info_ratios = bench_info_power[bench_id][simpoint_id]
      for version_iter in bench_info_power[bench_id][simpoint_id]:
        if version_iter <= 0:
            print(version_iter + ' is 0 ,error')


  [int_ratio, fp_ratio, bp_ratio, load_ratio, store_ratio] = get_inst_radio(bench_id, simpoint_id)

  ideal = float(bench_info_ratios[4])
  exe = ideal / float(bench_info_ratios[1]) 
  dcache = ideal / float(bench_info_ratios[2])
  frontend = ideal / float(bench_info_ratios[3])
  bench_info_transform = [0, exe, dcache, frontend]

  if (random.random() < (bench_info_transform[1])):
    exe_int = int(random.random() * 2) #max2
  else:
    exe_int = 0
  if (random.random() < (bench_info_transform[1] * fp_ratio)):
    exe_fp = int(random.random() * 2) #max2
  else:
    exe_fp = 0
  if (random.random() < (bench_info_transform[1])):
    lsq = int(random.random() * 4) #max4
  else:
    lsq = 1
  if (random.random() < (bench_info_transform[2])):
    dcache = int(random.random() * 3) #max3
  else:
    dcache = 0
  if (random.random() < (bench_info_transform[2])):
    l2cache = int(random.random() * 2) #max2
  else:
    l2cache = 1
  if (random.random() < (bench_info_transform[3])):
    icache = int(random.random() * 3) #max3
  else:
    icache = 0
  if (random.random() < (bench_info_transform[3])):
    bp = int(random.random() * 2) #max2
  else:
    bp = 1
  version_iter = gen_version_choose(DISPATCH_WIDTH_index, exe_int, exe_fp, lsq, dcache, icache, bp, l2cache)
  return version_iter

def gen_version_choose(DISPATCH_WIDTH_index, exe_int, exe_fp, lsq, dcache, icache, bp, l2cache):
  version = ['0' for i in range(int(version_map_id['max']))]
  #DISPATCH_WIDTH = DISPATCH_WIDTH_map[version[version_map_id['DISPATCH_WIDTH'][0]]]
  version[version_map_id['DISPATCH_WIDTH'][0]] = str(DISPATCH_WIDTH_index)
  version[version_map_id['IFQ_SIZE'][0]] = str(int(DISPATCH_WIDTH_index/2))
  version[version_map_id['DECODEQ_SIZE'][0]] = str(int(DISPATCH_WIDTH_index/2))
  version[version_map_id['FETCH_WIDTH'][0]] = str(DISPATCH_WIDTH_index)
  version[version_map_id['DECODE_WIDTH'][0]] = str(DISPATCH_WIDTH_index)
  version[version_map_id['COMMIT_WIDTH'][0]] = str(DISPATCH_WIDTH_index)
  version[version_map_id['PHY_GPR_NUM'][0]] = str(DISPATCH_WIDTH_index)
  version[version_map_id['PHY_FGPR_NUM'][0]] = str(DISPATCH_WIDTH_index)

  version[version_map_id['GPR_WRITEBACK_WIDTH'][0]] = str(int(DISPATCH_WIDTH_index/2))
  version[version_map_id['FGPR_WRITEBACK_WIDTH'][0]] = str(int(DISPATCH_WIDTH_index/2))
  version[version_map_id['RUU_SIZE_MAX'][0]] = str(DISPATCH_WIDTH_index)

  if -1 < exe_int:
    version[version_map_id['INT_BP'][0]] = str(exe_int)
    version[version_map_id['INT_ALU'][0]] = str(exe_int)
    version[version_map_id['INT_MULT'][0]] = str(exe_int)
    version[version_map_id['INT_MULT_OP_LAT'][0]] = str(exe_int)
    version[version_map_id['INT_MULT_ISSUE_LAT'][0]] = str(exe_int)
    version[version_map_id['INT_DIV_OP_LAT'][0]] = str(exe_int)
    version[version_map_id['INT_DIV_ISSUE_LAT'][0]] = str(exe_int)
  else:
    version[version_map_id['INT_BP'][0]] = str(int(DISPATCH_WIDTH_index/2))
    version[version_map_id['INT_ALU'][0]] = str(int(DISPATCH_WIDTH_index/2))
    version[version_map_id['INT_MULT'][0]] = str(int(DISPATCH_WIDTH_index/2))
    version[version_map_id['INT_MULT_OP_LAT'][0]] = str(int(DISPATCH_WIDTH_index/2))
    version[version_map_id['INT_MULT_ISSUE_LAT'][0]] = str(int(DISPATCH_WIDTH_index/2))  
    version[version_map_id['INT_DIV_OP_LAT'][0]] = str(int(DISPATCH_WIDTH_index/2))
    version[version_map_id['INT_DIV_ISSUE_LAT'][0]] = str(int(DISPATCH_WIDTH_index/2))  

  if -1 < exe_fp:
    version[version_map_id['FP_ALU'][0]] = str(exe_fp)
    version[version_map_id['FP_ALU_MULT'][0]] = str(exe_fp)
    version[version_map_id['FP_MULT_DIV'][0]] = str(exe_fp)
    version[version_map_id['FP_ALU_MULT_DIV'][0]] = str(exe_fp)
    version[version_map_id['FP_MULT_OP_LAT'][0]] = str(exe_fp)
    version[version_map_id['FP_MULT_ISSUE_LAT'][0]] = str(exe_fp)
    version[version_map_id['FP_DIV_OP_LAT'][0]] = str(exe_fp)
    version[version_map_id['FP_DIV_ISSUE_LAT'][0]] = str(exe_fp)
  else:
    version[version_map_id['FP_ALU'][0]] = str(int(DISPATCH_WIDTH_index/2))
    version[version_map_id['FP_ALU_MULT'][0]] = str(int(DISPATCH_WIDTH_index/2))
    version[version_map_id['FP_MULT_DIV'][0]] = str(int(DISPATCH_WIDTH_index/2))
    version[version_map_id['FP_ALU_MULT_DIV'][0]] = str(int(DISPATCH_WIDTH_index/2))
    version[version_map_id['FP_MULT_OP_LAT'][0]] = str(int(DISPATCH_WIDTH_index/2))  
    version[version_map_id['FP_MULT_ISSUE_LAT'][0]] = str(int(DISPATCH_WIDTH_index/2))
    version[version_map_id['FP_DIV_OP_LAT'][0]] = str(int(DISPATCH_WIDTH_index/2))
    version[version_map_id['FP_DIV_ISSUE_LAT'][0]] = str(int(DISPATCH_WIDTH_index/2))           

  if -1 < lsq:
    version[version_map_id['LOAD_PORT_WIDTH'][0]] = str(int(lsq/2))
    version[version_map_id['STORE_PORT_WIDTH'][0]] = str(int(lsq/2))
    version[version_map_id['LOAD_STORE_PORT_WIDTH'][0]] = str(int(lsq/2))
    version[version_map_id['LOAD_QUEUE_SIZE'][0]] = str(lsq)
    version[version_map_id['STORE_QUEUE_SIZE'][0]] = str(lsq)
  else:
    version[version_map_id['LOAD_PORT_WIDTH'][0]] = str(int(DISPATCH_WIDTH_index/2))
    version[version_map_id['STORE_PORT_WIDTH'][0]] = str(int(DISPATCH_WIDTH_index/2))
    version[version_map_id['LOAD_STORE_PORT_WIDTH'][0]] = str(int(DISPATCH_WIDTH_index/2))
    version[version_map_id['LOAD_QUEUE_SIZE'][0]] = str(DISPATCH_WIDTH_index)
    version[version_map_id['STORE_QUEUE_SIZE'][0]] = str(DISPATCH_WIDTH_index)

  if -1 < bp:
    version[version_map_id['BPRED'][0]] = str(bp)
    version[version_map_id['RAS_SIZE'][0]] = str(bp)
  else:
    version[version_map_id['BPRED'][0]] = str(int(DISPATCH_WIDTH_index/2))
    version[version_map_id['RAS_SIZE'][0]] = str(int(DISPATCH_WIDTH_index/2))

  if -1 < icache:
    version[version_map_id['L1_ICACHE_SET'][0]] = str(icache)
    version[version_map_id['L1_ICACHE_ASSOC'][0]] = str(icache)
  else:
    version[version_map_id['L1_ICACHE_SET'][0]] = str(int((1 + DISPATCH_WIDTH_index) / 2))
    version[version_map_id['L1_ICACHE_ASSOC'][0]] = str(int((1 + DISPATCH_WIDTH_index) / 2))

  if -1 < dcache:
    version[version_map_id['L1_DCACHE_SET'][0]] = str(dcache)
    version[version_map_id['L1_DCACHE_ASSOC'][0]] = str(dcache)
    version[version_map_id['L1_DCACHE_WRITEBACK'][0]] = str(int(dcache/2))
  else:
    version[version_map_id['L1_DCACHE_SET'][0]] = str(int((1 + DISPATCH_WIDTH_index) / 2))
    version[version_map_id['L1_DCACHE_ASSOC'][0]] = str(int((1 + DISPATCH_WIDTH_index) / 2))
    version[version_map_id['L1_DCACHE_WRITEBACK'][0]] = str(int((DISPATCH_WIDTH_index) / 2))

  if -1 < l2cache:
    version[version_map_id['L2_CACHE_SET'][0]] = str(l2cache)
    version[version_map_id['L2_CACHE_ASSOC'][0]] = str(l2cache)
  else:
    version[version_map_id['L2_CACHE_SET'][0]] = str(int(DISPATCH_WIDTH_index/2))
    version[version_map_id['L2_CACHE_ASSOC'][0]] = str(int(DISPATCH_WIDTH_index/2))

  version[version_map_id['LLC'][0]] = '0'

  version_str = ''
  for version_iter in version:
    version_str += version_iter
  return version_str

def data_loader(data_path):

    #configs = read_config('./config_history/')
    #configs = read_config('./config_history_all/')
    #read_area(configs)
    configs = configs_all
    case_length = len(configs[0]['params'])
    #print(configs[0]['params'])

    if bench_indivisual_model is None or simpoint_mode:
        bench_soft_info_length = 2 * len(specific_version)  #1 for bench tag(one hot coding)
        if inst_radio_mode:
            bench_soft_info_length += 5 #inst_radio length
        print('bench_soft_info_length = ' + str(bench_soft_info_length))
        log_file.write('bench_soft_info_length = ' + str(bench_soft_info_length) + '\n')
    else:
        bench_soft_info_length = 0
    
    #data_path = './data_all/'
    case_num_all = 0
    #case_num = 0
    final_data = []
    final_cpi_labels = []
    final_power_labels = []
    final_labels = []
    for bench_file in os.listdir(data_path):
        if bench_indivisual_model is not None and bench_indivisual_model not in bench_file:
            continue
        data_file = open(data_path + bench_file, "r", encoding='utf-8')
        bench_id = 0
        bench_name_str = bench_file.split('-')
        bench_name = bench_name_str[0]
        simpoint_id = 0
        if (2 < len(bench_name_str)):
            simpoint_id = int(bench_name_str[2].split('.')[0])
        for bench_name_cmp in bench_array:
            if bench_name_cmp[1] in bench_name:
                break
            bench_id += 1
        if len(bench_array) < bench_id:
            print(bench_name +' not found')
            continue
        #print(bench_file)
        data = data_file.read()
        data_file.close()
        row_list = data.splitlines()
        case_num_all += len(row_list)
        data_list = [[i for i in row.strip().split(" ")] for row in row_list]
        #print("data_list", data_list)
        [bench_cases, cpi_label, power_label, bench_labels, cpi_range, power_range] = str2value(bench_id, simpoint_id, data_list, configs)
        #case_num += len(bench_cases)
        inst_radio = get_inst_radio(bench_id, simpoint_id)
        global bench_info
        bench_info[bench_id][simpoint_id] = {
            'case_num': len(bench_cases),
            'cpi_range' : cpi_range,
            'power_range' : power_range,
            'inst_radio' : inst_radio
        }
        for case_iter, case, cpi, power, label in zip(range(len(bench_cases)), bench_cases, cpi_label, power_label, bench_labels):
            hw_sf_data = []
            case_ignore = 0
            #hw_sf_data += bench_info_cpi[bench_id][simpoint_id]
            #hw_sf_data += bench_info_power[bench_id][simpoint_id]

            for version_iter in bench_info_cpi[bench_id][simpoint_id]:
                if version_iter < 0:
                    case_ignore = 1
                    break
                else:
                    if bench_indivisual_model is None or simpoint_mode:
                        hw_sf_data.append(version_iter)

            for version_iter in bench_info_power[bench_id][simpoint_id]:
                if version_iter < 0:
                    case_ignore = 1
                    break
                else:
                    if bench_indivisual_model is None or simpoint_mode:
                        hw_sf_data.append(version_iter)                    

            if case_ignore:
                break
            else:
                if inst_radio_mode:
                    hw_sf_data += inst_radio
                input_transform(hw_sf_data, case)
                final_data.append(hw_sf_data)
                final_labels.append(label)
                final_cpi_labels.append(cpi)
                final_power_labels.append(power)
                #print(bench_file, bench_labels[case_iter][CASE_VERSION_INDEX], 'collect')
                #log_file.write(bench_file + ' ' + bench_labels[case_iter][CASE_VERSION_INDEX] + ' collect\n')

    print('case_length = ' + str(case_length))
    log_file.write('case_length = ' + str(case_length) + '\n')            
    #case_length_real  = case_length - len(input_mask_array)
    if input_enable:
        case_length_real = len(input_enable_array)
    else:
        case_length_real = case_length - len(input_mask_array)
        #case_length_real = case_length
    #case_length_real = len(bench_cases[0])
    print('case_length_real = ' + str(case_length_real))
    log_file.write('case_length_real = ' + str(case_length_real) + '\n')

    print('data cases num all = ' + str(case_num_all))
    log_file.write('data cases num all = ' + str(case_num_all) + '\n')
    assert(0 < case_num_all)
    case_num = len(final_data)
    print('data cases num = ' + str(case_num))
    log_file.write('data cases num = ' + str(case_num) + '\n')
    assert(0 < case_num)
    output_bench_info()
    input_length = case_length_real + bench_soft_info_length
    #final_data = np.array(final_data)
    #final_data = torch.Tensor(final_data)
    return [final_data, final_cpi_labels, final_power_labels, final_labels, bench_info, case_num, input_length]

'''
def input_transform(hw_sf_data, case):
    for value_index,case_data in zip(range(len(case)), case):
        if input_enable:
            input_enable_match = 0            
            for input_enable_element in input_enable_array:
                if input_enable_element == value_index:
                    input_enable_match = 1
                    break
            if input_enable_match:
                if DEF_L1_ICACHE_SET == value_index or DEF_L1_DCACHE_SET == value_index or DEF_L2_CACHE_SET == value_index:
                    hw_sf_data.append(math.log(case_data, 2))
                else:
                    hw_sf_data.append(case_data)
        else:
            if DEF_L1_ICACHE_SET == value_index or DEF_L1_DCACHE_SET == value_index or DEF_L2_CACHE_SET == value_index:
                hw_sf_data.append(math.log(case_data, 2))
            else:
                hw_sf_data.append(case_data)            
'''

def input_transform(hw_sf_data, case):
    for value_index,case_data in zip(range(len(case)), case):
        try:
            if DEF_L1_ICACHE_SET == value_index or DEF_L1_DCACHE_SET == value_index or DEF_L2_CACHE_SET == value_index:
                hw_sf_data.append(math.log(case_data, 2))
            elif DEF_FREQ == value_index:
                hw_sf_data.append(case_data/1000000000)
            else:
                input_mask_match = 0
                for input_mask_element in input_mask_array:
                    if input_mask_element == value_index:
                        input_mask_match = 1
                        break
                if 0 == input_mask_match:
                    hw_sf_data.append(case_data)
        except:
            print(value_index, case_data, 'except')

def str2value(bench_id, simpoint_id, data, configs):
    cpi_range = [999.0, 0.0]
    power_range = [999.0, 0.0]
    #if 56 == bench_id:
    #	print("bench_id: " + str(bench_id))
    #col_len = len(data)
    row_len = len(data[0])
    raw_data = []
    cpi_label = []
    power_label = []
    raw_label = []
    for row in data:
        #row_data = []
        config_version = row[0]
        config = get_config(configs, config_version)
        if config is not None:
            config_vector = config['params']
        else:
            #print(config_version + ' unfound ignore')
            config_vector = None
            #continue
        statistics = {
            'cpi' : float(row[1]),
            'bpred' : float(row[2]),
            'core_power' : float(row[3]),
            'weight_sum' : float(row[4]),
        }
        if area_mode:
            try:
                if 0 < config['area']:
                    statistics['core_power'] = config['area']
                else:
                    print(config_version + ' area is 0, ignore')
                    continue
            except:
                print(config_version + ' area not find')
        #过滤无效case
        if (config_vector is not None):
            #print(row)
            row_data = copy.deepcopy(config_vector)
            row_label = []
            #row_data.append(1 << bench_id)
            #for j in range(1, row_len-2):
            #    row_data.append(int(row[j]))
            if 0 == simpoint_id and statistics['weight_sum'] < 0.97:
                    print('error bench: ', bench_array[bench_id][1], config_version, statistics['weight_sum'], 'weight_sum < 0.97')
                    continue                
            #if CPI_mode:
            if 0 < statistics['cpi']:
                cpi_label.append(statistics['cpi']) #cpi(target)
            else:
                print('error bench: ', bench_array[bench_id][1], statistics['cpi'], 'cpi < 0')
                exit(1)
                continue
            cpi_range = [min(cpi_range[0], statistics['cpi']), max(cpi_range[1], statistics['cpi'])]	                
            #else:
            if 0 < statistics['core_power']:
                power_label.append(statistics['core_power'])
            else:
                print('error bench: ', bench_array[bench_id][1], statistics['core_power'], 'core_power < 0')
                exit(1)
                continue
            power_range = [min(power_range[0], statistics['core_power']), max(power_range[1], statistics['core_power'])]
            #print("row_data", row_data, 'len', len(row_data))
            #row_data.append(float(row[row_len-1])) #bpred
            raw_data.append(row_data)
            # fixed togother with BENCH_ID_INDEX , CASE_VERSION_INDEX
            row_label.append(bench_id)
            row_label.append(simpoint_id)
            row_label.append(config_version)
            raw_label.append(row_label)
        for version_id,config_version_cmp in zip(range(0,len(specific_version)), specific_version):
            if config_version_cmp in config_version:
                global bench_info_cpi
                bench_info_cpi[bench_id][simpoint_id][version_id] = statistics['cpi']
                global bench_info_power
                bench_info_power[bench_id][simpoint_id][version_id] = statistics['core_power']
    #print("row_data", row_data)
    #data_normalization(raw_data, cpi_range)
    return [raw_data, cpi_label, power_label, raw_label, cpi_range, power_range]

def data_normalization(data, cpi_range):
    cpi_index = len(data[0])-1
    for row in data:
        row[cpi_index] /= cpi_range[1]
        #row[cpi_index] *= 100


def data_preprocess(raw_data, feature_range = (0,1)):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range, copy=True)
    min_max_scaler.fit(raw_data)
    final_data = min_max_scaler.transform(raw_data)
    return [final_data, min_max_scaler]

def data_postprocess(final_data, min_value, max_value):
    #row_data = min_max_scaler.inverse_transform(final_data)
    row_data = final_data * (max_value - min_value) + min_value
    return row_data

def get_inst_radio(bench_id, simpoint_id):
    #int, fp, bp, load, store
    #stats.txt 484 line
    if 11 == bench_id:
        #403.2 gcc simpoint-1
        inst_radio = [4313627 / 10000000, 0 / 10000000, 2392038 /10000000, 2344198 / 10000000, 1404120 / 10000000]
    elif 32 == bench_id:
        #471.1 omnetpp simpoint-1
        inst_radio = [3752684 / 10000000, 254 / 10000000, 3730542 /10000000, 1494782 / 10000000, 1021825 / 10000000]
    else:
        #473.2 astar  simpoint-6
        #inst_radio = [4854549 / 10000000, 0 / 10000000, 1900767 /10000000, 2848886 / 10000000, 395799 / 10000000]
        print('fix inst_radio')
    return inst_radio

def output_bench_info():
    bench_info = open('bench_info.txt', 'w')
    bench_info.write('bench_name' + ' ')
    for version in specific_version:
        bench_info.write(version + ' ')
    bench_info.write('\n')
    for bench_cpi_cases,bench_power_cases,bench_cases_name in zip(bench_info_cpi, bench_info_power, bench_array):
        bench_info.write(bench_cases_name[1] + ' ')
        #for simpoint_id, bench_case_simpoint in enumerate(bench_cpi_cases):
        for bench_case in bench_cpi_cases[simpoint_id]:
            bench_info.write(' ' + str(bench_case))
            #bench_cases[0] =
        #for simpoint_id, bench_case_simpoint in enumerate(bench_power_cases):
        for bench_case in bench_power_cases[simpoint_id]:
            bench_info.write(' ' + str(bench_case))
        bench_info.write('\n')
    bench_info.close()

def shuffle(list1, list2, list3, list4):
    shuffled_list1 = []
    shuffled_list2 = [] 
    shuffled_list3 = []
    shuffled_list4 = []
    list_Len = len(list1)
    row_index = [x for x in range(list_Len)]
    random.shuffle(row_index)
    for i in range(list_Len):
        shuffled_list1.append(list1[row_index[i]])
        shuffled_list2.append(list2[row_index[i]])
        shuffled_list3.append(list3[row_index[i]])
        shuffled_list4.append(list4[row_index[i]])        
    return [shuffled_list1, shuffled_list2, shuffled_list3, shuffled_list4]

class MLP_Predictor(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_size, drop_rate, use_bias, use_drop, initial_lr, momentum, loss_fun, bench_each_norm, batch_size, CPI_mode):
        super(MLP_Predictor, self).__init__()
        #hidden_size_1 = hidden_size
        #hidden_size_2 = 
        self.layer1 = nn.Linear(in_channel, hidden_size, bias = use_bias)
        self.layer2 = nn.Linear(hidden_size, hidden_size, bias = use_bias)
        self.layer3 = nn.Linear(hidden_size, out_channel, bias = use_bias)
        #self.layer4 = nn.Linear(hidden_size, out_channel, bias = use_bias)
        self.hidden_size = hidden_size
        self.use_drop = use_drop
        self.dropout = nn.Dropout(drop_rate)
        #lr scheduler  每25个epoch drop 0.25

        #self.optimizer = optimizer
        self.optimizer = torch.optim.SGD(self.parameters(), lr = initial_lr, momentum = momentum)
        #self.optimizer = torch.optim.Adam(self.parameters(), lr = initial_lr)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 2000, gamma=1.5, last_epoch=-1)       
        self.loss_fun = loss_fun
        self.bench_each_norm = bench_each_norm
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.batch_size = batch_size
        self.CPI_mode = CPI_mode

    def forward(self, _input, input_length):
        x = _input #[0: input_length]
        x = torch.FloatTensor(x)
        #print("layer0: x", x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # print("layer2 x", x)
        #x = F.relu(self.layer3(x))
        #x = F.sigmoid(x)
        x = self.layer3(x)
        x = torch.sigmoid(x)
        #x = self.layer4(x) 
        if self.use_drop:
            y = self.dropout(x)
        else:
            y = x
        #print("y", y)
        return y

    def forward_batch(self, _input, batch_size):
        x = _input
        x = torch.FloatTensor(x)
        #print("layer0: x", x)
        x_1 = self.layer1(x)
        x_1 = self.bn1(x_1)

        #if self.use_drop:
        #    y = self.dropout(x_1)
        #else:
        #    y = x_1

        x_1 = F.relu(x_1)

        x_2 = self.layer2(x_1)
        x_2 = self.bn2(x_2)        
        x_2 = F.relu(x_2)

        x = self.layer3(x_2)
        #x_3 = self.bn3(x_3)
        #x_3 = F.relu(x_3)
        # print("layer2 x", x)
        #x = F.relu(self.layer3(x))
        #x = F.sigmoid(x)

        if self.use_drop:
            x = self.dropout(x)
        else:
            x = x

        #x_4 = self.layer4(x_3)
        y = torch.sigmoid(x)
        #x_3 = F.relu(x_3)

        #print("y", y)
        return y

    def my_train_top(self, train_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value, batch_normalization):
        if batch_normalization:
            return self.my_train_batch(train_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value)
        else:
            return self.my_train(train_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value)

    def my_train(self, train_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value):
        #data_shuffle(batch_input)
        self.optimizer.zero_grad()
        train_loss = 0
        for case_iter in range(batch_number):
            output_nn = self.forward(train_data[case_iter], input_length)
            #output = torch.sigmoid(output)
            simpoint_id = real_y_label[case_iter][BENCH_SIMPOINT_INDEX]
            if self.bench_each_norm:
                if self.CPI_mode:
                    range_info = bench_info[real_y_label[case_iter][BENCH_ID_INDEX]][simpoint_id][0]['cpi_range']
                else:
                    range_info = bench_info[real_y_label[case_iter][BENCH_ID_INDEX]][simpoint_id][0]['power_range']
                min_value = range_info[0]
                max_value = range_info[1]
            output = data_postprocess(output_nn, min_value, max_value)
            loss = self.loss_fun(torch.as_tensor(output, dtype = torch.float32), torch.as_tensor(label[case_iter], dtype = torch.float32))
            #loss = self.loss_fun(output, torch.as_tensor(real_y_label[case_iter][0]))
            #loss = loss_fun(output.to(torch.float32), y_label)
            #loss = loss_fun(output, y_label) / pow(y_label,2)
            #loss = (1 - (output / y_label))
            #if 0 == (epoch_number % print_peroid):
            #print("y_label", real_y_label[case_iter][0], ", output", output)
            #print(model.layer1.weight)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / batch_number
        return train_loss

    def my_train_batch(self, train_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value):
        #data_shuffle(batch_input)

        train_loss = 0
        batch_size = self.batch_size
        for b in range(int(batch_number/batch_size)):
            self.optimizer.zero_grad()
            y_lable_batch = torch.Tensor([x for x in label[b*batch_size:(b+1)*batch_size]])
            train_data_batch = train_data[b*batch_size:(b+1)*batch_size]
            loss_sum = 0
            output = self.forward_batch(train_data_batch, batch_size)
            #output = torch.sigmoid(output)
            if self.bench_each_norm:
                min_value = []
                max_value = []
                for case_iter in range(batch_size):
                    simpoint_id = real_y_label[b*batch_size+case_iter][BENCH_SIMPOINT_INDEX]
                    if self.CPI_mode:
                        range_info = bench_info[real_y_label[b*batch_size+case_iter][BENCH_ID_INDEX]][simpoint_id]['cpi_range']
                    else:
                        range_info = bench_info[real_y_label[b*batch_size+case_iter][BENCH_ID_INDEX]][simpoint_id]['power_range']
                    min_value.append(range_info[0])
                    max_value.append(range_info[1])
                min_value = torch.Tensor(min_value)
                max_value = torch.Tensor(max_value)
                #loss = self.loss_fun(torch.as_tensor(output, dtype = torch.float32), torch.as_tensor(real_y_label[b][0], dtype = torch.float32))
                #loss = self.loss_fun(output[case_iter], y_lable_batch[case_iter])
                #loss_sum += loss
            #output_2 = torch.squeeze(output)
            output_real = data_postprocess(output.flatten(), min_value, max_value)
            loss = self.loss_fun(output_real.flatten(), y_lable_batch.flatten())
            loss_sum = loss.sum()
            #loss = loss_fun(output.to(torch.float32), y_label)
            #loss = loss_fun(output, y_label) / pow(y_label,2)
            #loss = (1 - (output / y_label))
            # if 0 == (epoch_number % print_peroid):
                # print('epoch', epoch_number, "y_label", y_label, ", output", output)
            #print(model.layer1.weight)
            loss_sum.backward()
            self.optimizer.step()
            train_loss += loss_sum.item()
            #print(output_real, y_lable_batch,loss_sum, '!')

        #self.scheduler.step()
        train_loss = train_loss / batch_number
        return train_loss

    def my_eval_top(self, eval_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value, batch_normalization, train_final):
        if batch_normalization:
            return self.my_eval_batch(eval_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value, train_final)
        else:
            return self.my_eval(eval_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value, train_final)

    def my_eval(self, eval_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value, train_final):
        eval_loss = 0
        error_max = 0
        for b in range(batch_number):
            output = self.forward(eval_data[b], input_length)
            #output = torch.sigmoid(output)
            if self.bench_each_norm:
                if self.CPI_mode:
                    range_info = bench_info[real_y_label[b][BENCH_ID_INDEX]][0]['cpi_range']
                else:
                    range_info = bench_info[real_y_label[b][BENCH_ID_INDEX]][0]['power_range']
                min_value = range_info[0]
                max_value = range_info[1]
            output = data_postprocess(output, min_value, max_value)
            loss = self.loss_fun(torch.as_tensor(output, dtype = torch.float32), torch.as_tensor(label[b], dtype = torch.float32))
            #loss = loss_fun(output.to(torch.float32), y_label)
            #loss = loss_fun(output, y_label) / pow(y_label,2)
            #loss = (1 - (output / y_label))
            # if 0 == (epoch_number % print_peroid):
                # print('epoch', epoch_number, "y_label", y_label, ", output", output)
            #print(model.layer1.weight)
            error_max = max(error_max, loss)
            eval_loss += loss.item()
            #print(output,real_y_label[b][0] ,loss, '@')

        #print('error_max=', error_max)
        eval_loss = eval_loss / batch_number
        return [eval_loss, error_max]

    def my_eval_batch(self, eval_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value, train_final):
        output = self.forward_batch(eval_data, batch_number)
        label_tensor = torch.FloatTensor(label)
        #label_tensor = torch.Tensor([x for x in label])
        if self.bench_each_norm:
            min_value = []
            max_value = []
            for case_iter in range(batch_number):
                simpoint_id = real_y_label[case_iter][BENCH_SIMPOINT_INDEX]
                if self.CPI_mode:
                    range_info = bench_info[real_y_label[case_iter][BENCH_ID_INDEX]][simpoint_id]['cpi_range']
                else:
                    range_info = bench_info[real_y_label[case_iter][BENCH_ID_INDEX]][simpoint_id]['power_range']
                min_value.append(range_info[0])
                max_value.append(range_info[1])
            min_value = torch.Tensor(min_value)
            max_value = torch.Tensor(max_value)
            #loss = self.loss_fun(torch.as_tensor(output, dtype = torch.float32), torch.as_tensor(real_y_label[b][0], dtype = torch.float32))
            #loss = self.loss_fun(output[case_iter], y_lable_batch[case_iter])
            #loss_sum += loss
        #output_2 = torch.squeeze(output)
        output_real = data_postprocess(output.flatten(), min_value, max_value)
        match_num = 0
        [train_cpi_data, train_power_data, train_y_label] = train_final
        for output_real_iter,real_y_label_iter in enumerate(real_y_label):
            for train_cpi,train_power,train_label in zip(train_cpi_data, train_power_data, train_y_label):
                if real_y_label_iter[CASE_VERSION_INDEX] == train_label[CASE_VERSION_INDEX]:
                    output_real[output_real_iter] = label[output_real_iter]
                    match_num += 1
                    break
        #print('eval match train_num=' + str(match_num))
        if 0:
            loss = self.loss_fun(output_real.flatten(), label_tensor.flatten())
        else:
            loss_model = Loss_Fun()
            loss = loss_model(output_real.flatten(), label_tensor.flatten())
        error_max = max(loss)
        #print(loss.shape)
        #error_max = 20
        #print('error_max=', error_max)
        loss_sum = loss.sum().item()

        eval_loss = loss_sum / batch_number
        return [eval_loss, error_max]

    def cdf(self, eval_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value, error_interval, error_max, opt):
        intervals = int(error_max/error_interval) + 1
        error_cdf = [0 for x in range(intervals)]
        label_tensor = torch.FloatTensor(label)
        output = self.forward_batch(eval_data, batch_number)
        #real_y_label_cpi = torch.Tensor([x[0] for x in label])
        if self.bench_each_norm:
            min_value = []
            max_value = []
            for case_iter in range(batch_number):
                simpoint_id = real_y_label[case_iter][BENCH_SIMPOINT_INDEX]
                if self.CPI_mode:
                    range_info = bench_info[real_y_label[case_iter][BENCH_ID_INDEX]][simpoint_id]['cpi_range']
                else:
                    range_info = bench_info[real_y_label[case_iter][BENCH_ID_INDEX]][simpoint_id]['power_range']
                min_value.append(range_info[0])
                max_value.append(range_info[1])
            min_value = torch.Tensor(min_value)
            max_value = torch.Tensor(max_value)
            #loss = self.loss_fun(torch.as_tensor(output, dtype = torch.float32), torch.as_tensor(real_y_label[b][0], dtype = torch.float32))
            #loss = self.loss_fun(output[case_iter], y_lable_batch[case_iter])
            #loss_sum += loss
        #output_2 = torch.squeeze(output)
        output_real = data_postprocess(output.flatten(), min_value, max_value)
        loss = self.loss_fun(output_real.flatten(), label_tensor.flatten())
        
        #error_vector[int(loss/error_interval)-1] += 1
            #print(output,real_y_label[b][0] ,loss, '@')
        error_cdf[0] = 0
        for interval_id in range(1, intervals):
            #error_percent.append(interval_id * error_interval)
            for loss_i in loss:
                if loss_i < interval_id * error_interval:
                    error_cdf[interval_id] += 1 / batch_number
        #error_cdf[-1] = 1
        error_limit = error_cdf[int(0.05 / error_interval)]
        if self.CPI_mode:
            CPI_mode_str = 'CPI'
        else:
            CPI_mode_str = 'power'
        cdf_file = open('log/' + CPI_mode_str + '-' + str(sample_num) + opt + '-cdf.txt', 'w')
        for interval_id,cdf_err in zip(range(intervals), error_cdf):
            cdf_file.write(str(interval_id) + ' ' + str(interval_id * error_interval) + ' '+ str(cdf_err) + ' \n')
        cdf_file.close()

        print(CPI_mode_str + ' 5\% coverage error_limit=' + str(error_limit))
        log_file.write(CPI_mode_str + ' 5\% coverage error_limit=' + str(error_limit) + '\n')
        return error_cdf

class MLP_Predictor_2(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_size, drop_rate, use_bias, use_drop, initial_lr, momentum, loss_fun, bench_each_norm, batch_size, CPI_mode):
        super(MLP_Predictor_2, self).__init__()
        #hidden_size_1 = hidden_size
        #hidden_size_2 = 
        self.layer1 = nn.Linear(in_channel, hidden_size*2, bias = use_bias)
        self.layer2 = nn.Linear(hidden_size*2, hidden_size*2, bias = use_bias)
        self.layer3 = nn.Linear(hidden_size*2, hidden_size, bias = use_bias)
        self.layer4 = nn.Linear(hidden_size, out_channel, bias = use_bias)
        self.hidden_size = hidden_size
        self.use_drop = use_drop
        self.dropout = nn.Dropout(drop_rate)
        #lr scheduler  每25个epoch drop 0.25

        #self.optimizer = optimizer
        self.optimizer = torch.optim.SGD(self.parameters(), lr = initial_lr, momentum = momentum)
        #self.optimizer = torch.optim.Adam(self.parameters(), lr = initial_lr)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 2000, gamma=1.5, last_epoch=-1)       
        self.loss_fun = loss_fun
        self.bench_each_norm = bench_each_norm
        self.bn1 = nn.BatchNorm1d(hidden_size*2)
        self.bn2 = nn.BatchNorm1d(hidden_size*2)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.batch_size = batch_size
        self.CPI_mode = CPI_mode

    def forward(self, _input, input_length):
        x = _input #[0: input_length]
        x = torch.FloatTensor(x)
        #print("layer0: x", x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # print("layer2 x", x)
        #x = F.relu(self.layer3(x))
        #x = F.sigmoid(x)
        x = self.layer3(x)
        x = torch.sigmoid(x)
        #x = self.layer4(x) 
        if self.use_drop:
            y = self.dropout(x)
        else:
            y = x
        #print("y", y)
        return y

    def forward_batch(self, _input, batch_size):
        x = _input
        x = torch.FloatTensor(x)
        #print("layer0: x", x)
        x_1 = self.layer1(x)
        x_1 = self.bn1(x_1)

        if self.use_drop:
            x_1 = self.dropout(x_1)
        else:
            x_1 = x_1

        x_1 = F.relu(x_1)

        x_2 = self.layer2(x_1)
        x_2 = self.bn2(x_2)
        x_2 = F.relu(x_2)

        x_3 = self.layer3(x_2)
        x_3 = self.bn3(x_3)
        x_3 = F.relu(x_3)
        # print("layer2 x", x)
        #x = F.relu(self.layer3(x))
        #x = F.sigmoid(x)

        x = self.layer4(x_3)
        if self.use_drop:
            x = self.dropout(x)
        else:
            x = x

        y = torch.sigmoid(x)
        #x_3 = F.relu(x_3)

        #print("y", y)
        return y

    def my_train_top(self, train_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value, batch_normalization):
        if batch_normalization:
            return self.my_train_batch(train_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value)
        else:
            return self.my_train(train_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value)

    def my_train(self, train_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value):
        #data_shuffle(batch_input)
        self.optimizer.zero_grad()
        train_loss = 0
        for case_iter in range(batch_number):
            output_nn = self.forward(train_data[case_iter], input_length)
            #output = torch.sigmoid(output)
            simpoint_id = real_y_label[case_iter][BENCH_SIMPOINT_INDEX]
            if self.bench_each_norm:
                if self.CPI_mode:
                    range_info = bench_info[real_y_label[case_iter][BENCH_ID_INDEX]][simpoint_id][0]['cpi_range']
                else:
                    range_info = bench_info[real_y_label[case_iter][BENCH_ID_INDEX]][simpoint_id][0]['power_range']
                min_value = range_info[0]
                max_value = range_info[1]
            output = data_postprocess(output_nn, min_value, max_value)
            loss = self.loss_fun(torch.as_tensor(output, dtype = torch.float32), torch.as_tensor(label[case_iter], dtype = torch.float32))
            #loss = self.loss_fun(output, torch.as_tensor(real_y_label[case_iter][0]))
            #loss = loss_fun(output.to(torch.float32), y_label)
            #loss = loss_fun(output, y_label) / pow(y_label,2)
            #loss = (1 - (output / y_label))
            #if 0 == (epoch_number % print_peroid):
            #print("y_label", real_y_label[case_iter][0], ", output", output)
            #print(model.layer1.weight)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / batch_number
        return train_loss

    def my_train_batch(self, train_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value):
        #data_shuffle(batch_input)

        train_loss = 0
        batch_size = self.batch_size
        for b in range(int(batch_number/batch_size)):
            self.optimizer.zero_grad()
            y_lable_batch = torch.Tensor([x for x in label[b*batch_size:(b+1)*batch_size]])
            train_data_batch = train_data[b*batch_size:(b+1)*batch_size]
            loss_sum = 0
            output = self.forward_batch(train_data_batch, batch_size)
            #output = torch.sigmoid(output)
            if self.bench_each_norm:
                min_value = []
                max_value = []
                for case_iter in range(batch_size):
                    simpoint_id = real_y_label[b*batch_size+case_iter][BENCH_SIMPOINT_INDEX]
                    if self.CPI_mode:
                        range_info = bench_info[real_y_label[b*batch_size+case_iter][BENCH_ID_INDEX]][simpoint_id]['cpi_range']
                    else:
                        range_info = bench_info[real_y_label[b*batch_size+case_iter][BENCH_ID_INDEX]][simpoint_id]['power_range']
                    min_value.append(range_info[0])
                    max_value.append(range_info[1])
                min_value = torch.Tensor(min_value)
                max_value = torch.Tensor(max_value)
                #loss = self.loss_fun(torch.as_tensor(output, dtype = torch.float32), torch.as_tensor(real_y_label[b][0], dtype = torch.float32))
                #loss = self.loss_fun(output[case_iter], y_lable_batch[case_iter])
                #loss_sum += loss
            #output_2 = torch.squeeze(output)
            output_real = data_postprocess(output.flatten(), min_value, max_value)
            loss = self.loss_fun(output_real.flatten(), y_lable_batch.flatten())
            loss_sum = loss.sum()
            #loss = loss_fun(output.to(torch.float32), y_label)
            #loss = loss_fun(output, y_label) / pow(y_label,2)
            #loss = (1 - (output / y_label))
            # if 0 == (epoch_number % print_peroid):
                # print('epoch', epoch_number, "y_label", y_label, ", output", output)
            #print(model.layer1.weight)
            loss_sum.backward()
            self.optimizer.step()
            train_loss += loss_sum.item()
            #print(output_real, y_lable_batch,loss_sum, '!')

        #self.scheduler.step()
        train_loss = train_loss / batch_number
        return train_loss

    def my_eval_top(self, eval_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value, batch_normalization, train_final):
        if batch_normalization:
            return self.my_eval_batch(eval_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value, train_final)
        else:
            return self.my_eval(eval_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value, train_final)
                        
    def my_eval(self, eval_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value, train_final):
        eval_loss = 0
        error_max = 0
        for b in range(batch_number):
            output = self.forward(eval_data[b], input_length)
            #output = torch.sigmoid(output)
            if self.bench_each_norm:
                if self.CPI_mode:
                    range_info = bench_info[real_y_label[b][BENCH_ID_INDEX]][0]['cpi_range']
                else:
                    range_info = bench_info[real_y_label[b][BENCH_ID_INDEX]][0]['power_range']
                min_value = range_info[0]
                max_value = range_info[1]
            output = data_postprocess(output, min_value, max_value)
            loss = self.loss_fun(torch.as_tensor(output, dtype = torch.float32), torch.as_tensor(label[b], dtype = torch.float32))
            #loss = loss_fun(output.to(torch.float32), y_label)
            #loss = loss_fun(output, y_label) / pow(y_label,2)
            #loss = (1 - (output / y_label))
            # if 0 == (epoch_number % print_peroid):
                # print('epoch', epoch_number, "y_label", y_label, ", output", output)
            #print(model.layer1.weight)
            error_max = max(error_max, loss)
            eval_loss += loss.item()
            #print(output,real_y_label[b][0] ,loss, '@')

        #print('error_max=', error_max)
        eval_loss = eval_loss / batch_number
        return [eval_loss, error_max]

    def my_eval_batch(self, eval_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value, train_final):
        output = self.forward_batch(eval_data, batch_number)
        label_tensor = torch.FloatTensor(label)
        #label_tensor = torch.Tensor([x for x in label])
        if self.bench_each_norm:
            min_value = []
            max_value = []
            for case_iter in range(batch_number):
                simpoint_id = real_y_label[case_iter][BENCH_SIMPOINT_INDEX]
                if self.CPI_mode:
                    range_info = bench_info[real_y_label[case_iter][BENCH_ID_INDEX]][simpoint_id]['cpi_range']
                else:
                    range_info = bench_info[real_y_label[case_iter][BENCH_ID_INDEX]][simpoint_id]['power_range']
                min_value.append(range_info[0])
                max_value.append(range_info[1])
            min_value = torch.Tensor(min_value)
            max_value = torch.Tensor(max_value)
            #loss = self.loss_fun(torch.as_tensor(output, dtype = torch.float32), torch.as_tensor(real_y_label[b][0], dtype = torch.float32))
            #loss = self.loss_fun(output[case_iter], y_lable_batch[case_iter])
            #loss_sum += loss
        #output_2 = torch.squeeze(output)
        output_real = data_postprocess(output.flatten(), min_value, max_value)
        [train_cpi_data, train_power_data, train_y_label] = train_final
        for output_real_iter,real_y_label_iter in enumerate(real_y_label):
            for train_cpi,train_power,train_label in zip(train_cpi_data, train_power_data, train_y_label):
                if real_y_label_iter[CASE_VERSION_INDEX] == train_label[CASE_VERSION_INDEX]:
                    output_real[output_real_iter] = label[output_real_iter]
                    break

        if 0:
            loss = self.loss_fun(output_real.flatten(), label_tensor.flatten())
        else:
            loss_model = Loss_Fun()
            loss = loss_model(output_real.flatten(), label_tensor.flatten())
        error_max = max(loss)
        #print(loss.shape)
        #error_max = 20
        #print('error_max=', error_max)
        loss_sum = loss.sum().item()

        eval_loss = loss_sum / batch_number
        return [eval_loss, error_max]

    def cdf(self, eval_data, input_length, label, real_y_label, batch_number, bench_info, min_value, max_value, error_interval, error_max, opt):
        intervals = int(error_max/error_interval) + 1
        error_cdf = [0 for x in range(intervals)]
        label_tensor = torch.FloatTensor(label)
        output = self.forward_batch(eval_data, batch_number)
        #real_y_label_cpi = torch.Tensor([x[0] for x in label])
        if self.bench_each_norm:
            min_value = []
            max_value = []
            for case_iter in range(batch_number):
                simpoint_id = real_y_label[case_iter][BENCH_SIMPOINT_INDEX]
                if self.CPI_mode:
                    range_info = bench_info[real_y_label[case_iter][BENCH_ID_INDEX]][simpoint_id]['cpi_range']
                else:
                    range_info = bench_info[real_y_label[case_iter][BENCH_ID_INDEX]][simpoint_id]['power_range']
                min_value.append(range_info[0])
                max_value.append(range_info[1])
            min_value = torch.Tensor(min_value)
            max_value = torch.Tensor(max_value)
            #loss = self.loss_fun(torch.as_tensor(output, dtype = torch.float32), torch.as_tensor(real_y_label[b][0], dtype = torch.float32))
            #loss = self.loss_fun(output[case_iter], y_lable_batch[case_iter])
            #loss_sum += loss
        #output_2 = torch.squeeze(output)
        output_real = data_postprocess(output.flatten(), min_value, max_value)
        loss = self.loss_fun(output_real.flatten(), label_tensor.flatten())
        
        #error_vector[int(loss/error_interval)-1] += 1
            #print(output,real_y_label[b][0] ,loss, '@')
        error_cdf[0] = 0
        for interval_id in range(1, intervals):
            #error_percent.append(interval_id * error_interval)
            for loss_i in loss:
                if loss_i < interval_id * error_interval:
                    error_cdf[interval_id] += 1 / batch_number
        #error_cdf[-1] = 1
        error_limit = error_cdf[int(0.05 / error_interval)]
        if self.CPI_mode:
            CPI_mode_str = 'CPI'
        else:
            CPI_mode_str = 'power'
        cdf_file = open('log/' + CPI_mode_str + '-' + str(sample_num) + opt + '-cdf.txt', 'w')
        for interval_id,cdf_err in zip(range(intervals), error_cdf):
            cdf_file.write(str(interval_id) + ' ' + str(interval_id * error_interval) + ' '+ str(cdf_err) + ' \n')
        cdf_file.close()            
        print(CPI_mode_str + ' 5\% coverage error_limit=' + str(error_limit))
        log_file.write(CPI_mode_str + ' 5\% coverage error_limit=' + str(error_limit) + '\n')
        return error_cdf

def write_error(final, sample_id, bench_id, simpoint_id, cpi_dsp, power_dsp, version_dsp, opt):
    global sample_num    
    [final_cpi_labels, final_power_labels, real_y_label_array] = final
    fig_name_str = 'error_sample/' + str(bench_id) + '-' + str(simpoint_id) + '-sample-' + str(sample_num) + '-' + str(opt) + '-error.txt'
    file = open(fig_name_str, 'w')
    if opt == '_opt':
        opt_str = 'iDSE'
    else:
        opt_str = 'baseline'
    cpi_hit = []
    power_hit = []
    version_hit = []
    for cpi,power,version_iter in zip(cpi_dsp, power_dsp, version_dsp):
        for real_cpi, real_power, real_y in zip(final_cpi_labels, final_power_labels, real_y_label_array):
            if real_y[CASE_VERSION_INDEX] == version_iter['name']:
                file.write(str(sample_num) + ' ' + opt_str + ' ' + str(cpi/real_cpi-1) + ' ' + str(power/real_power-1) + '\n')
                cpi_hit.append(cpi)
                power_hit.append(power)
                version_hit.append(version_iter['name'])
                break
    file.close()
    return [cpi_hit, power_hit, version_hit]

def pareto_optimality(models, final, pareto_points, sample_id, train_final, opt, area_mode):
    if bench_indivisual_model is not None:
        bench_id = 0
        for bench_name_cmp in bench_array:
            if bench_name_cmp[1] in bench_indivisual_model:
                break
            bench_id += 1
        if len(bench_array) < bench_id:
            print(bench_name +' not found')
    else:
        bench_id = 11 #403.2
    global simpoint_id
    hw_sf_data = []
    for version_iter in bench_info_cpi[bench_id][simpoint_id]:
        if 0 < version_iter:
            if bench_indivisual_model is None or simpoint_mode:
                hw_sf_data.append(version_iter)
    for version_iter in bench_info_power[bench_id][simpoint_id]:
        if 0 < version_iter:
            if bench_indivisual_model is None or simpoint_mode:
                hw_sf_data.append(version_iter)                
    if inst_radio_mode:
        hw_sf_data += get_inst_radio(bench_id, simpoint_id)

    data_dsp_raw = []
    data_area = []
    if 0:
        [pareto_points_x, pareto_points_y, points_config, sample_num_i] = pareto_points
        for config in points_config:
            data_row = copy.deepcopy(hw_sf_data)
            try:
                for config_match in configs_all:
                    if config_match['name'] == config:
                        input_transform(data_row, config_match['params'])
                        data_dsp_raw.append(data_row)
                        data_area.append(config_match['area'])
                        break
            except:
                print(config + 'unmatch')
    else:
        for config in configs_all:
            data_row = copy.deepcopy(hw_sf_data)
            input_transform(data_row, config['params'])
            data_dsp_raw.append(data_row)
            data_area.append(config['area'])

    [data_dsp, min_max_scaler] = data_preprocess(data_dsp_raw)
    input_length = len(data_dsp[0])
    batch_number = len(configs_all)
    max_value = min_value = 0
    batch_normalization = 1
    for model in models:
        model.eval()
    cpi_output = models[0].forward_batch(data_dsp, batch_number)
    if area_mode:
        power_output = data_area
    else:
        power_output = models[1].forward_batch(data_dsp, batch_number)

    if models[0].bench_each_norm:
        min_value_cpi = []
        max_value_cpi = []
        min_value_power = []
        max_value_power = []        
        for case_iter in range(batch_number):
            #simpoint_id = real_y_label[case_iter][BENCH_SIMPOINT_INDEX]
            range_info_cpi = bench_info[bench_id][simpoint_id]['cpi_range']
            range_info_power = bench_info[bench_id][simpoint_id]['power_range']
            min_value_cpi.append(range_info_cpi[0])
            max_value_cpi.append(range_info_cpi[1])
            min_value_power.append(range_info_power[0])
            max_value_power.append(range_info_power[1])
        min_value_cpi = torch.Tensor(min_value_cpi)
        max_value_cpi = torch.Tensor(max_value_cpi)
        min_value_power = torch.Tensor(min_value_power)
        max_value_power = torch.Tensor(max_value_power)        
        #loss = self.loss_fun(torch.as_tensor(output, dtype = torch.float32), torch.as_tensor(real_y_label[b][0], dtype = torch.float32))
        #loss = self.loss_fun(output[case_iter], y_lable_batch[case_iter])
        #loss_sum += loss
    #output_2 = torch.squeeze(output)
    cpi_dsp_x = data_postprocess(cpi_output.flatten(), min_value_cpi, max_value_cpi)
    cpi_dsp = [x.item() for x in cpi_dsp_x]

    if area_mode:
        power_dsp = power_output
    else:
        power_dsp = data_postprocess(power_output.flatten(), min_value_power, max_value_power)
        power_dsp = [x.item() for x in power_dsp]

    match_num = 0
    [train_cpi_data, train_power_data, train_y_label] = train_final
    for output_real_iter,real_y_label_iter in enumerate(configs_all):
        for train_cpi,train_power,train_label in zip(train_cpi_data, train_power_data, train_y_label):
            if real_y_label_iter['name'] == train_label[CASE_VERSION_INDEX]:
                cpi_dsp[output_real_iter] = train_cpi
                if 0 == area_mode:
                    power_dsp[output_real_iter] = train_power
                match_num += 1
                break
    print('pareto_optimality match_num=' + str(match_num))

    [cpi_hit, power_hit, version_hit] = write_error(final, sample_id, bench_id, simpoint_id, cpi_dsp, power_dsp, configs_all, opt)

    filename = get_new_add_dsp_filename(opt, simpoint_id, sample_id)
    [points_x, points_y, points_x_pareto, points_y_pareto, pareto_configs] = get_pareto_point(filename, cpi_hit, power_hit, version_hit, pareto_points, final, train_final, opt, mode = 0)
    return [points_x, points_y, points_x_pareto, points_y_pareto, pareto_configs]

def get_new_add_dsp_filename(opt, simpoint_id, sample_id):
    if area_mode:
        filename = 'dsp_cpi_area'
    else:
        filename = 'dsp_cpi_power'
    if bench_indivisual_model:
        filename += '_' + str(opt) + '_' + bench_indivisual_model + '-' + str(simpoint_id) + '-' + str(sample_id)
    filename += '.txt'
    return filename

def get_dsp_filename(opt, simpoint_id, sample_id):
    if area_mode:
        filename = 'log_dsp/all_dsp_cpi_area'
    else:
        filename = 'log_dsp/all_dsp_cpi_power'
    if bench_indivisual_model:
        filename += '_' + opt + '_' + bench_indivisual_model + '-' + str(simpoint_id) + '-' + str(sample_id)
    filename += '.txt'
    return filename    

def get_pareto_point(filename, cpi_dsp, power_dsp, configs, pareto_points, final, train_final, opt, mode):

    assert(len(cpi_dsp) == len(configs))
    assert(len(power_dsp) == len(configs))

    if 0 == mode:
        #[final_cpi_labels, final_power_labels, real_y_label_array] = final
        if 0 == demo:
            [pareto_points_x, pareto_points_y, points_config, sample_num_i] = pareto_points
        else:
            points_config = []
        pareto_points_hit_num = 0
        [train_cpi, train_power, train_y_label] = train_final

    #cpi_max = 10#max(cpi_dsp)
    #cpi_min = 0#min(cpi_dsp)

    if demo:
        intervals = 200
        boxes_size = 200
    else:
        intervals = 300
        boxes_size = 2

    value_min = 0    
    if area_mode:
        value_max = 40
        value_min = 5
        intervals = 50
        boxes_size = 2
    else:
        value_max = 20
    value_range_width = value_max - value_min

    value_interval = ((value_range_width) / intervals)
    dsp_set = [[] for x in range(intervals+1)]
    points_x = []
    points_y = []
    points_x_pareto = []
    points_y_pareto = []
    #points_config = []
    pareto_configs = []

    if demo:
        power_limit = 5
    else:
        power_limit = 100

    new_add_dsp_file = open(filename, 'w')
    new_add_dsp_file.write(str(sample_num) + '\n')
    dsp_filename = get_dsp_filename(opt, simpoint_id, sample_id)
    dsp_file = open(dsp_filename, 'w')
    for config, cpi_dsp_iter, power_dsp_iter in zip(configs, cpi_dsp, power_dsp):
        if power_dsp_iter < value_max:
            value_dsp_iter_box = int((power_dsp_iter-value_min) / value_interval)
            #if 1.75 < cpi_dsp_iter:
            #if 0 == demo and 3 < cpi_dsp_iter:
            #   continue
        else:
            continue
        #if power_dsp_iter.item() < power_limit:
        if mode:
            version_iter = config[CASE_VERSION_INDEX]
        else:
            version_iter = config #['name']
        if len(version_iter) < 30:
            continue
        heapq.heappush(dsp_set[value_dsp_iter_box], (0-cpi_dsp_iter, power_dsp_iter, version_iter))
        #print(dsp_set[value_dsp_iter_box])
        if boxes_size < len(dsp_set[value_dsp_iter_box]):
            heapq.heappop(dsp_set[value_dsp_iter_box])
        #print(dsp_set[value_dsp_iter_box])
    for interval_id in range(intervals):
        dsp_points = dsp_set[interval_id]
        #print('[' + str(interval_id*cpi_interval) + ',' + str((interval_id+1)*cpi_interval) + '] : ')
        for index in range(len(dsp_points)):
            dsp_point = heapq.heappop(dsp_points)
            #print(dsp_point)
            version_hit = dsp_point[2]
            if 0 == len(points_y_pareto) or (0 - dsp_point[0]) < points_y_pareto[-1]:
                str_output = str(0 - dsp_point[0]) + ' ' + str(dsp_point[1]) + ' ' + version_hit +'\n'   
                if 0 == mode:
                    has_found = False
                    for real_cpi, real_power, real_label in zip(train_cpi, train_power, train_y_label):
                        if real_label[CASE_VERSION_INDEX] == version_hit:
                            has_found = True
                            break                     
                    pareto_configs.append(version_hit)
                    if version_hit in points_config:
                        pareto_points_hit_num += 1
                    dsp_file.write(str_output)
                    if not has_found:
                        new_add_dsp_file.write(str_output)
                if 0 == len(dsp_points):
                    points_y_pareto.append(-dsp_point[0])
                    points_x_pareto.append(dsp_point[1])
                    #points_config.append(dsp_point[2])
                    if mode:
                        new_add_dsp_file.write(str_output)
                else:
                    if (-dsp_point[0]) < power_limit:
                        points_y.append(-dsp_point[0])
                        points_x.append(dsp_point[1])
    new_add_dsp_file.close()
    dsp_file.close()
    #cmd_str = 'cp ' + filename + '.txt' + ' choose/' + filename + '-' + str(sample_id) + '.txt'
    #os.system(cmd_str.decode('utf-8'))
    if 0 == mode:
        print(filename + ': pareto_points_hit_num/all = ' + str(pareto_points_hit_num) + ' / ' + str(len(points_config)) + ' \n')
        log_file.write(filename + ': pareto_points_hit_num/all = ' + str(pareto_points_hit_num) + ' / ' + str(len(points_config)) + ' \n')
        hit_file = open('log/hit.log', 'a')
        hit_file.write(filename + ' ' + str(sample_num) + ' : pareto_points_hit_num/all = ' + str(pareto_points_hit_num) + ' / ' + str(len(points_config)) + ' \n')
        hit_file.close()
    return [points_x, points_y, points_x_pareto, points_y_pareto, pareto_configs]

class Loss_Fun(nn.Module):
    def __init__(self):
        super(Loss_Fun, self).__init__()
        
    def forward(self, y, label):
        return torch.abs((y / label) - 1)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def hierarchical_cluster(data_M):
    #print(data_M)
    n_clusters = 5
    Z = linkage(data_M, method='ward', metric='euclidean')
    p = dendrogram(Z, 0)
    ac = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    #ac.fit(data_M)

    #AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
    #            connectivity=None, linkage='ward', memory=None, n_clusters=3)

    #聚类
    labels = ac.fit_predict(data_M)
    print("hierarchical_cluster n_clusters= " + str(n_clusters))
    print(labels)
    log_file.write("hierarchical_cluster n_clusters= " + str(n_clusters) + ':\n')
    for label in labels:
        log_file.write(str(label) + ' ')
    log_file.write('\n')

    #plt.scatter(data_M[:,0], data_M[:,1], c=labels)
    plt.gca().set(xlabel = 'benchmarks ', ylabel = 'linkage distance')
    #plt.title("benchmark hierarchy cluster ", fontsize=15)
    plt.savefig("cluster.png",  dpi=1000)
    plt.show()

def main():

    # 设置随机数种子
    setup_seed(4)

    startTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    startTime = datetime.strptime(startTime,"%Y-%m-%d %H:%M:%S")

    max_epoch_number = 20000
    print_peroid = 200
    loss_delta_thredhold = 0.0000000001
    #batch_number = 2 #有x行数据就写x
    bench_each_norm = 1 #not changeable now
    bench_choose_opt = 1
    batch_normalization = 1
    cdf_enable = 1

    cpi_mode_train = 1
    power_mode = 1
    train_loss_fig = 1

    if area_mode:
        train_loss_fig = 0
        cdf_enable = 0
        power_mode = 0

    if demo:
        max_epoch_number = 3000
        print_peroid = 1000
        bench_choose_opt = 1
        train_loss_fig = 0        
        cdf_enable = 0

    #[raw_data, final_cpi_labels, final_power_labels, real_y_label_array, bench_info, batch_number, input_length] = data_loader('./data_all/')
    [raw_data, final_cpi_labels, final_power_labels, real_y_label_array, bench_info, batch_number, input_length] = data_loader('./data_all_simpoint/')

    [all_input, min_max_scaler] = data_preprocess(raw_data)
    #real_y_label_array: cpi, bench_id, version
    [all_input, final_cpi_labels, final_power_labels, real_y_label_array] = shuffle(all_input, final_cpi_labels, final_power_labels, real_y_label_array)

    #hierarchical_cluster(bench_info_cpi[0:])

    if 0:
        [random_data, random_cpi_label, random_power_label, random_y_label, choose_case, choose_cpi_label, choose_power_label, choose_y_label] = choose_runned_case(all_input, final_cpi_labels, final_power_labels, real_y_label_array)
        train_data_size_opt = int(len(choose_y_label) * 0.8)
        train_data_opt = choose_case[0:train_data_size_opt]
        train_cpi_opt = choose_cpi_label[0:train_data_size_opt]
        train_power_opt = choose_power_label[0:train_data_size_opt]
        train_y_label_opt = choose_y_label[0:train_data_size_opt]
        train_data_size_opt = len(train_data_opt)

        train_data_size = int(len(random_data) * 0.8)
        #train_data_size = int(min(len(train_data_opt), len(random_data)) * 0.9)
        #train_data_size = len(random_data)
        train_data = random_data[0:train_data_size]
        train_cpi_data = random_cpi_label[0:train_data_size]
        train_power_data = random_power_label[0:train_data_size]
        train_y_label = random_y_label[0:train_data_size]

        eval_data_size = len(choose_y_label) + len(random_data) - train_data_size_opt - train_data_size
        #eval_data_size = train_data_size
        eval_data = random_data[train_data_size:] + choose_case[train_data_size_opt:]
        eval_cpi_data = random_cpi_label[train_data_size:] + choose_cpi_label[train_data_size_opt:]
        eval_power_data = random_power_label[train_data_size:] + choose_power_label[train_data_size_opt:]
        eval_y_label = random_y_label[train_data_size:] + choose_y_label[train_data_size_opt:]
        assert(eval_data_size == len(eval_data))
        
    else:
        #[all_input, real_y_label_array] = shuffle(all_input, real_y_label_array)
        global sample_id

        if bench_choose_opt:

            train_data_size_opt_limit = sample_id * 25
            if choose_from_dsp and 1 < sample_id:
                train_data_size_opt_limit = 1000 #sample_id * 25

            if area_mode:
                pareto_optimality_perfect_filename = 'area-pareto_optimality_perfect.txt'
            else:
                pareto_optimality_perfect_filename = 'power-pareto_optimality_perfect.txt'
            get_pareto_point(pareto_optimality_perfect_filename, final_cpi_labels, final_power_labels, real_y_label_array, None, None, None, '', mode = 1)
            [real_pareto_points_x, real_pareto_points_y, real_pareto_points_config, sample_num_i] = get_pareto_optimality_from_file(pareto_optimality_perfect_filename)
            print('##get real pareto_optimality done')

            if 1:
                [train_data_opt, train_cpi_opt, train_power_opt, train_y_label_opt] = choose_runned_case2(train_data_size_opt_limit, all_input, final_cpi_labels, final_power_labels, real_y_label_array, real_pareto_points_config)           
            else:
                train_data_opt = all_input[0:train_data_size_opt_limit]
                train_cpi_opt = final_cpi_labels[0:train_data_size_opt_limit]
                train_power_opt = final_power_labels[0:train_data_size_opt_limit]
                train_y_label_opt = real_y_label_array[0:train_data_size_opt_limit]
                config_random_file = open('config_random_file.txt', 'w')
                for label in train_y_label_opt:
                    config_random_file.write(label[CASE_VERSION_INDEX] + '\n')
                config_random_file.close()
            print('##get train data opt done') 

            if 0:
                dps_version_list = get_dsp_version_to_next(opt = '_opt', simpoint_id = 1)
                print('dsp list length=' + str(len(dps_version_list)))
                [train_data_opt_dsp, train_cpi_opt_dsp, train_power_opt_dsp, train_y_label_opt_dsp] = dsp_data_filer(dps_version_list, all_input, final_cpi_labels, final_power_labels, real_y_label_array)
                print('dsp point length=' + str(len(train_y_label_opt_dsp)))
                train_data_opt += train_data_opt_dsp
                train_cpi_opt += train_cpi_opt_dsp
                train_power_opt += train_power_opt_dsp
                train_y_label_opt += train_y_label_opt

            equal_sample_size = 0
            if equal_sample_size and len(train_cpi_opt) < train_data_size:
                addition_size = train_data_size - len(train_cpi_opt)
                train_data_opt += all_input[:addition_size]
                train_cpi_opt += final_cpi_labels[:addition_size]
                train_power_opt += final_power_labels[:addition_size]
                train_y_label_opt += real_y_label_array[:addition_size]
            train_data_size_opt = len(train_cpi_opt)
        	#train_data_opt = all_input[0:train_data_size_opt]
        	#train_y_label_opt = real_y_label_array[0:train_data_size_opt]
            #[raw_data_opt, real_y_label_array_opt] = choose_case(raw_data, real_y_label_array)
            train_final_opt = [train_cpi_opt, train_power_opt, train_y_label_opt]

        if 1 == sample_id:
            train_data_size = sample_id * 25
        elif bench_choose_opt:
            train_data_size = train_data_size_opt
        else:
            train_data_size = sample_id * 25

        train_data = all_input[0:train_data_size]
        train_cpi_data = final_cpi_labels[0:train_data_size]
        train_power_data = final_power_labels[0:train_data_size]        
        train_y_label = real_y_label_array[0:train_data_size]
        train_final = [train_cpi_data, train_power_data, train_y_label]
        train_y_label_hit_num = 0
        for train_y_label_iter in train_y_label:
            if train_y_label_iter in real_pareto_points_config:
                train_y_label_hit_num += 1
        log_file.write('train_y_label_hit_num= ' + str(train_y_label_hit_num) + '\n')
        print('train_y_label_hit_num= ' + str(train_y_label_hit_num))
        if area_mode:
            hit_file = open('log/' + 'area-' + 'hit_file_choose.txt', 'a')
        else:
            hit_file = open('log/' + 'power-' + 'hit_file_choose.txt', 'a')
        hit_file.write('train_y_label_hit_num = ' + str(train_y_label_hit_num) + ' / ' + str(train_data_size) + ' / ' + str(len(real_pareto_points_config)) + '\n')
        hit_file.close()        

        if 1:
            eval_data = []
            eval_y_label = []
            for case_config in real_pareto_points_config:
                for case_input,case_label in zip(all_input,real_y_label_array):
                    if case_config == case_label[CASE_VERSION_INDEX]:
                        eval_data.append(case_input)
                        eval_y_label.append(case_label)
            eval_data_size = len(eval_y_label)
            assert(eval_data_size == len(real_pareto_points_config))
            eval_cpi_data = real_pareto_points_y
            eval_power_data = real_pareto_points_x          
        elif 1:
            eval_data_size = batch_number           
            eval_data = all_input
            eval_cpi_data = final_cpi_labels
            eval_power_data = final_power_labels
            eval_y_label = real_y_label_array
        else:
            eval_data_size = batch_number - train_data_size
            eval_data = all_input[train_data_size:]
            eval_cpi_data = final_cpi_labels[train_data_size:]
            eval_power_data = final_power_labels[train_data_size:]
            eval_y_label = real_y_label_array[train_data_size:]
        #eval_data = all_input
        #eval_y_label = real_y_label_array

    global sample_num
    sample_num = train_data_size
    print('train_data_size = ' + str(train_data_size))
    log_file.write('train_data_size = ' + str(train_data_size) + '\n')
    if bench_choose_opt:
        print('train_data_size_opt = ' + str(train_data_size_opt))
        log_file.write('train_data_size_opt = ' + str(train_data_size_opt) + '\n')        
    print('eval_data_size = ' + str(eval_data_size))
    log_file.write('eval_data_size = ' + str(eval_data_size) + '\n')

    output_length = 1
    print('input_length = ' + str(input_length))
    log_file.write('input_length = ' + str(input_length) + '\n')
    log_file.write('output_length = ' + str(output_length) + '\n')

    min_value = max_value = 0
    #real_y_label_array = [data[input_length] for data in raw_data]
    #y_label_array = [data[0:2] for data in real_y_label_array]
    #min_value = min(y_label_array)
    #max_value = max(y_label_array)
    #if CPI_mode:
    #log_file.write('cpi range: ' + 'min ' + str(min_value[0]) + ', max ' + str(max_value[0]) + '\n')
    #else:
    #log_file.write('core_power range: ' + 'min ' + str(min_value[1]) + ', max ' + str(max_value[1]) + '\n')    

    #[y_label_array, min_max_scaler_label] = data_preprocess(y_label_array, feature_range=(0.000001, 1))

    #loss_fun = nn.MSELoss()

    model_cpi = MLP_Predictor(in_channel = input_length, out_channel = output_length, hidden_size = 16
        , drop_rate = 0.01, use_bias=True, use_drop=True
        , initial_lr = 0.00001
        , momentum = 0.4
        , loss_fun = Loss_Fun()
        , bench_each_norm = bench_each_norm
        , batch_size = 25 #batch_size
        , CPI_mode = 1
        )
    #assert(0 == sample_num%50)
    model_power = MLP_Predictor_2(in_channel = input_length, out_channel = output_length, hidden_size = 32 #256
        , drop_rate = 0.02, use_bias=True, use_drop=True
        , initial_lr = 0.0005
        , momentum = 0.4
        , loss_fun = Loss_Fun() #nn.MSELoss() #
        , bench_each_norm = bench_each_norm
        , batch_size = 25 #batch_size
        , CPI_mode = 0
        )
    #assert(0 == sample_num%50)
    models = [model_cpi, model_power]
    models_opt = copy.deepcopy(models)

    #train_loss_vector_set = []
    train_loss_vector = []
    train_loss_vector_power = []
    train_loss_vector_opt = []
    train_loss_vector_opt_power = []
    train_loss_vector_epoch = []
    train_loss_vector_epoch_opt = []
    #validation_loss_vector_set = []
    validation_loss_vector = []
    validation_loss_vector_power = []
    validation_loss_vector_opt = []
    validation_loss_vector_opt_power = []
    #validation_loss_vector_epoch = []

    epoch_opt_min = epoch_min = epoch_number = 0
    last_train_loss = 0
    last_train_loss_power = 0
    epoch_number_fig_scale = 1000
    min_eval_loss_opt = min_eval_loss = 999
    converage_epoch = 0
    print('cpi model --------------------------------------------------------------------------------------')

    if cpi_mode_train:
        for e in range(max_epoch_number):
            #[train_data, train_y_label] = shuffle(train_data, train_y_label)
            [train_data, train_cpi_data, train_power_data, train_y_label] = shuffle(train_data, train_cpi_data, train_power_data, train_y_label)
            models[0].train()
            my_train_loss = models[0].my_train_top(train_data, input_length, train_cpi_data, train_y_label, train_data_size, bench_info, min_value, max_value, batch_normalization)      

            if (bench_choose_opt):
                [train_data_opt, train_cpi_opt, train_power_opt, train_y_label_opt] = shuffle(train_data_opt, train_cpi_opt, train_power_opt, train_y_label_opt)
                models_opt[0].train()
                my_train_loss_opt = models_opt[0].my_train_top(train_data_opt, input_length, train_cpi_opt, train_y_label_opt, train_data_size_opt, bench_info, min_value, max_value, batch_normalization)

            loss_delta = my_train_loss - last_train_loss
            last_train_loss = my_train_loss
            #train_loss_vector_set.append(my_train_loss)
            if 0 == (epoch_number % print_peroid):
                #my_train_loss_interval_min = min(train_loss_vector_set)
                train_loss_vector_epoch.append(epoch_number / epoch_number_fig_scale)
                train_loss_vector.append(my_train_loss)
                log_file.write("Training Epoch " + str(e) +  ', cpi: '  + str(my_train_loss))
                print("Training Epoch " + str(e) +  ', cpi: '  + str(my_train_loss))
                if (bench_choose_opt):
                    train_loss_vector_epoch_opt.append(epoch_number / epoch_number_fig_scale)
                    train_loss_vector_opt.append(my_train_loss_opt)
                    log_file.write(', opt_Loss cpi: ' + str(my_train_loss_opt))
                    print('       , opt_Loss cpi: ' + str(my_train_loss_opt) + ' # ' + str(my_train_loss_opt/my_train_loss))
                #train_loss_vector_set = []
                #log_file.write('\n')

            models[0].eval()
            [my_eval_loss, error_max] = models[0].my_eval_top(eval_data, input_length, eval_cpi_data, eval_y_label, eval_data_size, bench_info, min_value, max_value, batch_normalization, train_final)
            if (bench_choose_opt):
                models_opt[0].eval()
                [my_eval_loss_opt, error_max_opt] = models_opt[0].my_eval_top(eval_data, input_length, eval_cpi_data, eval_y_label, eval_data_size, bench_info, min_value, max_value, batch_normalization, train_final_opt)
         	
            # if 0 == (epoch_number % print_peroid):
                # print("validation test:", 'y_label', test_y_label, 'test_output', test_output)
            #test_loss_pow_avg = test_loss_pow_sum / batch_number
            #validation_loss_vector_set.append(my_eval_loss)
            if 0 == (epoch_number % print_peroid):
                #test_loss_average = min(validation_loss_vector_set)
                log_file.write(", test: " + str(my_eval_loss))
                print("Training Epoch: " + str(e) + ", test: " + str(my_eval_loss))
                #validation_loss_vector_epoch.append(epoch_number * train_data_size_opt)
                validation_loss_vector.append(my_eval_loss)
                #validation_loss_vector_set = []
                if bench_choose_opt:
                  validation_loss_vector_opt.append(my_eval_loss_opt)
                  print("               , test opt: "
                    + str(my_eval_loss_opt) + ' # ' + str(my_eval_loss_opt/my_eval_loss))
                  log_file.write(", test opt: " + str(my_eval_loss_opt))
                log_file.write('\n')

            epoch_number += 1
            if abs(loss_delta) < loss_delta_thredhold:
                log_file.write('loss_delta ' + str(loss_delta) + '< ' + 'loss_delta_thredhold ' + str(loss_delta_thredhold) + '\n')
                if 200 < epoch_number:
                    break
            if min_eval_loss < my_eval_loss:
                converage_epoch += 1
            else:
                converage_epoch = 0
                epoch_min = epoch_number
                min_eval_loss = my_eval_loss
            if bench_choose_opt:
                if min_eval_loss_opt < my_eval_loss_opt:
                    converage_epoch += 1
                else:
                    converage_epoch = 0
                    epoch_opt_min = epoch_number            
                    min_eval_loss_opt = my_eval_loss_opt
            if 3000 < converage_epoch:
                print('3000 < ' + 'converage_epoch = ' + str(converage_epoch))
                log_file.write('3000 < ' + 'converage_epoch ' + str(converage_epoch) + '\n')
                log_file.write("###### testing sample_num: " + str(train_data_size))
                log_file.write(", Epoch: " + str(epoch_min) + ", loss: " + str(min_eval_loss))
                if bench_choose_opt:
                    log_file.write("###### opt sample_num: " + str(train_data_size_opt))
                    log_file.write("Epoch: " + str(epoch_opt_min) + ", loss: " + str(min_eval_loss_opt))
                log_file.write('\n')
                break

        endTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        endTime = datetime.strptime(endTime,"%Y-%m-%d %H:%M:%S")
        # 相减得到秒数
        time_used = endTime - startTime
        print('time_used:', time_used, 'epoch', epoch_number, 'validation loss', min_eval_loss)
        log_file.write(str(time_used) + '\n')
        log_file.write('CPI mode time_used: ' + str(time_used))
        log_file.write(', epoch ' + str(epoch_min) + ', validation loss ' + str(min_eval_loss))
        cpi_sample_log_file = open('log/cpi_sample_log_file.txt', 'a')
        cpi_sample_log_file.write(str(train_data_size) + ' ' + str(epoch_min) + ' ' + str(min_eval_loss))
        if bench_choose_opt:
            log_file.write(', opt epoch ' + str(epoch_opt_min) + ', opt validation loss ' + str(min_eval_loss_opt))
            cpi_sample_log_file.write(' ' + str(train_data_size_opt) + ' ' + str(epoch_opt_min) + ' ' + str(min_eval_loss_opt))
        log_file.write('\n\n')
        cpi_sample_log_file.write('\n')
        cpi_sample_log_file.flush()
        cpi_sample_log_file.close()

    print('power/area model --------------------------------------------------------------------------------------')
    startTime_power = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    startTime_power = datetime.strptime(startTime_power,"%Y-%m-%d %H:%M:%S")

    if power_mode:
        train_loss_vector_power_epoch = []
        train_loss_vector_power_epoch_opt = []

        epoch_opt_min = epoch_min = epoch_number = 0
        min_eval_loss_opt = min_eval_loss = 999
        converage_epoch = 0
        for e in range(max_epoch_number):
            #[train_data, train_y_label] = shuffle(train_data, train_y_label)
            models[1].train()
            my_train_loss_power = models[1].my_train_top(train_data, input_length, train_power_data, train_y_label, train_data_size, bench_info, min_value, max_value, batch_normalization)        

            if (bench_choose_opt):
                models_opt[1].train()
                my_train_loss_opt_power = models_opt[1].my_train_top(train_data_opt, input_length, train_power_opt, train_y_label_opt, train_data_size_opt, bench_info, min_value, max_value, batch_normalization)

            loss_delta_power = my_train_loss_power - last_train_loss_power
            last_train_loss_power = my_train_loss_power
            #train_loss_vector_set.append(my_train_loss)
            if 0 == (epoch_number % print_peroid):
                #my_train_loss_interval_min = min(train_loss_vector_set)
                train_loss_vector_power_epoch.append(epoch_number / epoch_number_fig_scale)
                train_loss_vector_power.append(my_train_loss_power)
                log_file.write("Training Epoch " + str(e) + ', power: '  + str(my_train_loss_power))
                print("Training Epoch " + str(e) + ' , power: '  + str(my_train_loss_power))
                if (bench_choose_opt):
                    train_loss_vector_power_epoch_opt.append(epoch_number / epoch_number_fig_scale)
                    train_loss_vector_opt_power.append(my_train_loss_opt_power)
                    log_file.write(' , opt_Loss power: ' + str(my_train_loss_opt_power))
                    print(' , opt_Loss power: ' + str(my_train_loss_opt_power) + ' # ' + str(my_train_loss_opt_power/my_train_loss_power))
                    #print('delta: ', loss_delta_power)
                #log_file.write('\n')
                #train_loss_vector_set = []

            models[1].eval()
            [my_eval_loss_power, error_max_power] = models[1].my_eval_top(eval_data, input_length, eval_power_data, eval_y_label, eval_data_size, bench_info, min_value, max_value, batch_normalization, train_final)
            if (bench_choose_opt):
                models_opt[1].eval()
                [my_eval_loss_opt_power, error_max_opt_power] = models_opt[1].my_eval_top(eval_data, input_length, eval_power_data, eval_y_label, eval_data_size, bench_info, min_value, max_value, batch_normalization, train_final_opt)
            
            # if 0 == (epoch_number % print_peroid):
                # print("validation test:", 'y_label', test_y_label, 'test_output', test_output)
            #test_loss_pow_avg = test_loss_pow_sum / batch_number
            #validation_loss_vector_set.append(my_eval_loss)
            if 0 == (epoch_number % print_peroid):
                #test_loss_average = min(validation_loss_vector_set)
                log_file.write(", test: " + str(my_eval_loss_power))
                print("Training Epoch: " + str(e) + " , test: " + str(my_eval_loss_power))
                #validation_loss_vector_epoch.append(epoch_number * train_data_size_opt)
                validation_loss_vector_power.append(my_eval_loss_power)
                #validation_loss_vector_set = []
                if bench_choose_opt:
                  validation_loss_vector_opt_power.append(my_eval_loss_opt_power)
                  print("               , test opt: "
                    + str(my_eval_loss_opt_power) + ' # ' + str(my_eval_loss_opt_power/my_eval_loss_power))
                  log_file.write(" , test opt: " + str(my_eval_loss_opt_power))
                log_file.write('\n')

            epoch_number += 1
            if abs(loss_delta_power) < loss_delta_thredhold:
                log_file.write('loss_delta_power ' + str(loss_delta_power) + ' < ' + 'loss_delta_thredhold ' + str(loss_delta_thredhold) + '\n')
                if 200 < epoch_number:
                    break
            if min_eval_loss < my_eval_loss_power:
                converage_epoch += 1
            else:
                converage_epoch = 0
                epoch_min = epoch_number
                min_eval_loss = my_eval_loss_power
            if bench_choose_opt:
                if min_eval_loss_opt < my_eval_loss_opt_power:
                    converage_epoch += 1
                else:
                    converage_epoch = 0
                    epoch_opt_min = epoch_number
                    min_eval_loss_opt = my_eval_loss_opt_power
            if 3000 < converage_epoch:
                print('3000 < ' + 'converage_epoch ' + str(converage_epoch))
                log_file.write('1000 < ' + 'converage_epoch ' + str(converage_epoch) + '\n')
                log_file.write("###### testing sample_num: " + str(train_data_size))
                log_file.write(" , Epoch: " + str(epoch_min) + " , loss: " + str(min_eval_loss))
                if bench_choose_opt:
                    log_file.write(" ###### opt sample_num: " + str(train_data_size_opt))
                    log_file.write(" Epoch: " + str(epoch_opt_min) + " , loss: " + str(min_eval_loss_opt))
                log_file.write('\n')
                break

        endTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        endTime = datetime.strptime(endTime,"%Y-%m-%d %H:%M:%S")
        # 相减得到秒数
        time_used = endTime - startTime_power
        print('power mode time_used:', time_used, 'epoch', epoch_number, 'validation loss', min_eval_loss)
        #log_file.write(str(time_used))
        log_file.write('power mode time_used: ' + str(time_used))
        log_file.write(', epoch ' + str(epoch_min) + ', validation loss ' + str(min_eval_loss))
        power_sample_log_file = open('log/power_sample_log_file.txt', 'a')
        power_sample_log_file.write(str(train_data_size) + ' ' + str(epoch_min) + ' ' + str(min_eval_loss))
        if bench_choose_opt:
            log_file.write(', opt epoch ' + str(epoch_opt_min) + ' , opt validation loss ' + str(min_eval_loss_opt))
            power_sample_log_file.write(' ' + str(train_data_size_opt) + ' ' + str(epoch_opt_min) + ' ' + str(min_eval_loss_opt))
        log_file.write('\n')
        power_sample_log_file.write('\n')
        power_sample_log_file.flush()
        power_sample_log_file.close()

    fig_name = 'fig/loss_'
    if bench_indivisual_model:
        fig_name += bench_indivisual_model + '_' + 'sample-' + str(sample_num)
    
    fig_name_cpi = fig_name + 'cpi'
    
    fig_name_power = fig_name + 'core-power'
    #fig_name += '_cases-' + str(batch_number)

    #cdf_enable = 1
    if demo:
        cdf_enable = 0
    if cdf_enable:
        error_interval = 0.01
        error_max = 0.1
        if bench_choose_opt:
            error_max = max(error_max, error_max_opt)
            if power_mode:
                error_max_power = max(error_max_power, error_max_opt_power)
        error_max_power = error_max = 0.5

        intervals = int(error_max/error_interval) + 1
        error_percent = [(id * error_interval) for id in range(intervals)]
        intervals_power = int(error_max_power/error_interval) + 1
        error_percent_power = [(id * error_interval) for id in range(intervals_power)]

        for model in models:
            model.eval()
        error_cdf_cpi = models[0].cdf(eval_data, input_length, eval_cpi_data, eval_y_label, eval_data_size, bench_info, min_value, max_value, error_interval, error_max, '_')
        if power_mode:
            error_cdf_power = models[1].cdf(eval_data, input_length, eval_power_data, eval_y_label, eval_data_size, bench_info, min_value, max_value, error_interval, error_max_power, '_')
        if (bench_choose_opt):
            for model in models_opt:
                model.eval()    
            error_cdf_opt_cpi = models_opt[0].cdf(eval_data, input_length, eval_cpi_data, eval_y_label, eval_data_size, bench_info, min_value, max_value, error_interval, error_max, '_opt')
            if power_mode:
                error_cdf_opt_power = models_opt[1].cdf(eval_data, input_length, eval_power_data, eval_y_label, eval_data_size, bench_info, min_value, max_value, error_interval, error_max_power, '_opt')

        if 0:
            if bench_choose_opt:
                plt_cdf(error_percent, error_cdf_cpi, error_percent, error_cdf_opt_cpi, fig_name_cpi +'-cdf.pdf', 1)
                if power_mode:
                    plt_cdf(error_percent_power, error_cdf_power, error_percent_power, error_cdf_opt_power, fig_name_power +'-cdf.pdf', 0)
            else:
                plt_cdf(error_percent, error_cdf_cpi, [], [],  fig_name_cpi +'-cdf.pdf', 1)
                if power_mode:
                    plt_cdf(error_percent_power, error_cdf_power, [], [],  fig_name_power +'-cdf.pdf', 0)

    train_loss_fig = 0
    if train_loss_fig:
        if bench_choose_opt:
            if cpi_mode_train:
                output_fig(train_loss_vector_epoch, train_loss_vector
                    , train_loss_vector_epoch, validation_loss_vector
                    , train_loss_vector_epoch_opt, train_loss_vector_opt
                    , train_loss_vector_epoch_opt, validation_loss_vector_opt
                    , fig_name_cpi +'.pdf', 0, 1)
            if power_mode:
                output_fig(train_loss_vector_power_epoch, train_loss_vector_power
                , train_loss_vector_power_epoch, validation_loss_vector_power
                , train_loss_vector_power_epoch_opt, train_loss_vector_opt_power
                , train_loss_vector_power_epoch_opt, validation_loss_vector_opt_power      
                , fig_name_power +'.pdf', 0, 0)      
        else:
          output_fig(train_loss_vector_epoch, train_loss_vector
        	, train_loss_vector_epoch, validation_loss_vector
    		, [], []
    		, [], []
        	, fig_name_cpi +'.pdf', 0, 1)
          if power_mode:
              output_fig(train_loss_vector_epoch, train_loss_vector_power
                , train_loss_vector_epoch, validation_loss_vector_power
                , [], []
                , [], []        
                , fig_name_power +'.pdf', 0, 0)      
        #output_fig(validation_loss_vector_epoch, validation_loss_vector, 'predictor.png', 50)

    perfect = 1
    if demo:
        perfect = 0
    if perfect:
        if area_mode:
            pareto_optimality_perfect_filename = 'area-pareto_optimality_perfect.txt'
        else:
            pareto_optimality_perfect_filename = 'power-pareto_optimality_perfect.txt'
        #get_pareto_point(pareto_optimality_perfect_filename, final_cpi_labels, final_power_labels, real_y_label_array, None, None, None, '', mode = 1)
        pareto_points = get_pareto_optimality_from_file(pareto_optimality_perfect_filename)
    else:
        #pareto_points = [points_x, points_y, []]
        pareto_points = None
    print('#get_pareto_optimality_from_file perfect done')

    final = [final_cpi_labels, final_power_labels, real_y_label_array]
    baseline_points = pareto_optimality(models, final, pareto_points, sample_id, train_final, opt = '_', area_mode = area_mode)
    if (bench_choose_opt):
        opt_points = pareto_optimality(models_opt, final, pareto_points, sample_id, train_final_opt, opt = '_opt', area_mode = area_mode)

    if 0:
        if (bench_choose_opt):
            plot_pareto_optimality(baseline_points
                , opt_points
                , pareto_points)
        else:
            plot_pareto_optimality(baseline_points, [], pareto_points)

    endTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    endTime = datetime.strptime(endTime,"%Y-%m-%d %H:%M:%S")
    # 相减得到秒数
    time_used = endTime - startTime
    #print('all_time_used:', time_used, 'epoch', epoch_number, 'validation loss', min_eval_loss)
    #log_file.write(str(time_used))
    log_file.write('all_time_used: ' + str(time_used))
    time_log = open('log/time.log', 'a')
    time_log.write(str(sample_num) + ' ' + str(time_used) + ' \n')
    time_log.close()


def get_pareto_optimality_from_file(filename):
    file = open(filename, 'r')
    pareto_points_x = []
    pareto_points_y = []
    points_config = []
    sample_num_i = 0
    for point in file:
        point_str = point.split(' ')
        if 1 < len(point_str):
            cpi = float(point_str[0])
            if area_mode and 2.4 < cpi:
                continue
            pareto_points_y.append(cpi)
            pareto_points_x.append(float(point_str[1]))
            points_config.append(point_str[2].strip('\n'))
        elif 1 == len(point_str):
            sample_num_i = int(point.strip('\n'))
    file.close()
    return [pareto_points_x, pareto_points_y, points_config, sample_num_i]

#base_color = 'yellow'
#iDSE_color = 'gray'
demo = 0
if demo:
    dpi_set = 1000    
base_color = 'gray'
iDSE_color = 'black'
dpi_set = 100

def get_pareto_curve(points_x_pareto_opt, points_y_pareto_opt):
    curve_y = [points_y_pareto_opt[0]]
    for interval_id in range(1, len(points_x_pareto_opt)):
        curve_y.append(min(points_y_pareto_opt[0:interval_id+1]))
    return curve_y

def plot_pareto_optimality(baseline_points
    , opt_points
    , pareto_points
    ):

    [points_x_base, points_y_base, configs_base] = get_dsp_point(opt = '_', simpoint_id = simpoint_id)
    [points_x_opt, points_y_opt, configs_opt] = get_dsp_point(opt = '_opt', simpoint_id = simpoint_id)

    if area_mode:
        pareto_optimality_perfect_filename = 'area-pareto_optimality_perfect.txt'
    else:
        pareto_optimality_perfect_filename = 'power-pareto_optimality_perfect.txt'
    #get_pareto_point(pareto_optimality_perfect_filename, final_cpi_labels, final_power_labels, real_y_label_array, None, None, None, '', mode = 1)
    pareto_points = get_pareto_optimality_from_file(pareto_optimality_perfect_filename)

    #[points_x, points_y, points_x_pareto, points_y_pareto, sample_num_i] = baseline_points
    #if opt_points:
    #    [points_x_opt, points_y_opt, points_x_pareto_opt, points_y_pareto_opt, sample_num_i_opt] = opt_points

    if demo:
        base_color_dot = 'gray'
        base_color_line = 'black'
    else:
        base_color_dot = base_color
        base_color_line = base_color

    linestyles = ['dotted', 'dashed', '--']

    logbase_fig = 1
    if logbase_fig:
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        #ax.set_yscale('log')
        plt_handler = ax
    else:
        plt_handler = plt

    if 1:
        base_color_dot = '#BFBFBF'
        plt_handler.scatter(points_x_base, points_y_base, c = base_color_dot, s = 50, marker = 'o', label = 'random')
        #plt_handler.scatter(points_x_pareto, points_y_pareto, c = base_color_dot, s = 8, marker='o')
        #plt.plot(points_x, points_y, color=base_color, marker='o', markersize=5, label = 'candicate')
        #plt_handler.plot(points_x_pareto, points_y_pareto, color = base_color_line, marker='+', markersize=4, linestyle = linestyles[2], label = 'random')
    
    if points_x_opt:
        if demo:
            iDSE_color = 'gray'
            iDSE_color_line = 'black'
            iDSE_linestyle = None
        else:
            iDSE_color = 'black'
            iDSE_color_line = iDSE_color
            iDSE_linestyle = linestyles[1]
        #plt.scatter(points_x_opt, points_y_opt, c = iDSE_color, s = 14, marker='^')
        plt_handler.scatter(points_x_opt, points_y_opt, c = iDSE_color, s = 50, marker='^', label = 'iDSE')
        #curve_y = get_pareto_curve(points_x_pareto_opt, points_y_pareto_opt)
        #plt.plot(points_x, points_y, color=base_color, marker='o', markersize=5, label = 'candicate')
        #plt_handler.plot(points_x_opt, points_y_opt, color = iDSE_color_line, linestyle = iDSE_linestyle, label = 'iDSE')

    if pareto_points:
        real_coler = '#474747' #ColdGrey   DimGrey Grey    LightGrey SlateGrey   SlateGreyDark   SlateGreyLight  WarmGrey
        real_marker = '1' #'X'
        [pareto_points_x, pareto_points_y, pareto_points_config, sample_num_i_perfect] = pareto_points
        plt_handler.scatter(pareto_points_x, pareto_points_y, c = real_coler, s = 50, marker=real_marker, label = 'real')
        #plt.plot(points_x, points_y, color=base_color, marker='o', markersize=5, label = 'candicate')
        #real_curve_y = get_pareto_curve(pareto_points_x, pareto_points_y)
        #plt_handler.plot(pareto_points_x, pareto_points_y, color = real_coler, linestyle = linestyles[0], label = 'real')

    font1 = {'size' : 18}
    plt.ylabel('CPI', font = font1)
    if area_mode:
        mode_str = 'area-'
        plt.xlabel('Area(mm^2)', font = font1)        
        #plt.gca().set(xlabel = 'CPI', ylabel = 'Area(nm^2)')
        #plt.xlabel('CPI')
    else:
        mode_str = 'power-'
        plt.xlabel('Power(W)', font = font1)
        #plt.gca().set(xlabel = 'CPI', ylabel = 'Power(W)')
    #plt.title('pareto optimality in design space', fontsize=15)
    if demo:
        if area_mode:
            plt.xticks(np.arange(0, 50, 2))
        else:
            plt.xticks(np.arange(0, int(max(points_x_opt)+1), 1))
    else:
        plt.legend(fontsize = 17)
    #plt.grid(linestyle = '--')
    if demo:
        plt.savefig('pareto_optimality_demo.pdf', dpi=dpi_set)
    else:
        plt.savefig(bench_indivisual_model + '_' + mode_str + 'pareto_optimality-id' + str(sample_id) + '-' + str(sample_num) + '.pdf', dpi=dpi_set, format='pdf', bbox_inches='tight')
        #plt.savefig(mode_str + 'pareto_optimality.png', dpi=dpi_set)
    plt.show()

def plot_pareto_optimality_all():

    base_color_dots = ['black', '#5C5C5C', '#333333', '#1F1F1F', '#444444']
    base_color_markers = ['^', '.', '+', 'v', '*']
    linestyles = ['dashed', 'dotted', 'dashed', '--', '-']

    global sample_id
    for sample_id,base_color_dot,base_color_marker,each_linestyle in zip([sample_id_max], base_color_dots, base_color_markers, linestyles):
        filename = get_new_add_dsp_filename(opt = '_opt', simpoint_id = simpoint_id, sample_id = sample_id)
        [pareto_points_x, pareto_points_y, points_config, sample_num_i] = get_pareto_optimality_from_file(filename)
        plt.scatter(pareto_points_x, pareto_points_y, c = base_color_dot, s = 14, marker = base_color_marker)
        #plt.scatter(points_x_pareto, points_y_pareto, c = base_color_dot, s = 6, marker = base_color_marker)
        #plt.plot(points_x, points_y, color=base_color, marker='o', markersize=5, label = 'candicate')
        #curve_y = get_pareto_curve(pareto_points_x, pareto_points_y)
        plt.plot(pareto_points_x, pareto_points_y, color = base_color_dot, linestyle=each_linestyle, label = str(sample_num_i)) #)str(sample_id*10) + '%')
    
    if area_mode:
        pareto_optimality_perfect_filename = 'area-pareto_optimality_perfect.txt'
    else:
        pareto_optimality_perfect_filename = 'power-pareto_optimality_perfect.txt'
    #get_pareto_point(pareto_optimality_perfect_filename, final_cpi_labels, final_power_labels, real_y_label_array, '', mode = 1)
    [real_pareto_points_x, real_pareto_points_y, real_pareto_points_config, sample_num_i] = get_pareto_optimality_from_file(pareto_optimality_perfect_filename)

    real_coler = '#474747'
    real_marker = 'X'
    plt.scatter(real_pareto_points_x, real_pareto_points_y, c = real_coler, s = 14, marker=real_marker)
    #plt.plot(points_x, points_y, color=base_color, marker='o', markersize=5, label = 'candicate')
    #real_curve_y = get_pareto_curve(real_pareto_points_x, real_pareto_points_y)
    plt.plot(real_pareto_points_x, real_pareto_points_y, color = real_coler, linestyle = 'dotted', label = 'real')

    font1 = {'size' : 14}
    plt.ylabel('CPI', font = font1)
    if area_mode:
        mode_str = 'area-'
        plt.xlabel('Area(mm^2)', font = font1)
        #plt.gca().set(xlabel = 'CPI', ylabel = 'Area(nm^2)')
    else:
        mode_str = 'power-'
        plt.xlabel('Power(W)', font = font1)
        #plt.gca().set(xlabel = 'CPI', ylabel = 'Power(W)', fontsize = 15)
    #plt.title('pareto optimality in design space', fontsize=15)
    if demo:
        plt.xticks(np.arange(min(points_x), max(points_x), 0.5))
    else:
        plt.legend(fontsize = 14)
    plt.grid(linestyle = '--')
    if demo:
        plt.savefig('pareto_optimality_demo-sample.png', dpi=dpi_set)
    else:
        plt.savefig(mode_str + 'pareto_optimality-sample.pdf', dpi=dpi_set, format='pdf')
        #plt.savefig(mode_str + 'pareto_optimality.png', dpi=dpi_set)
    plt.show()

def output_fig(X_1, Y_1, X_2, Y_2, X_3, Y_3, X_4, Y_4, fig_name, epoch_print_min=0, CPI_mode = 1):
    epoch_print_min = 10
    if test_fig_mode:
        X_1_print = X_1[epoch_print_min:]
        Y_1_print = Y_1[epoch_print_min:]
        X_3_print = X_3[epoch_print_min:]
        Y_3_print = Y_3[epoch_print_min:]
    X_2_print = X_2[epoch_print_min:]
    Y_2_print = Y_2[epoch_print_min:]
    X_4_print = X_4[epoch_print_min:]
    Y_4_print = Y_4[epoch_print_min:]
    if test_fig_mode:
        plt.scatter(X_1_print, Y_1_print, c = base_color, s=5)
        plt.scatter(X_3_print, Y_3_print, c = iDSE_color, s=5)
    plt.scatter(X_2_print, Y_2_print, c = base_color, s=5)
    plt.scatter(X_4_print, Y_4_print, c = iDSE_color, s=5)
    if test_fig_mode:
        plt.plot(X_1_print, Y_1_print, color=base_color, marker='o', linestyle='--', markersize=2, label = 'train random')
        plt.plot(X_3_print, Y_3_print, color=iDSE_color, marker='+', linestyle='--', markersize=2, label = 'train iDSE')
    plt.plot(X_2_print, Y_2_print, color=base_color, marker='o', markersize=2, label = 'random')
    plt.plot(X_4_print, Y_4_print, color=iDSE_color, marker='+', markersize=2, label = 'iDSE')
    font1 = {'size' : 15}
    plt.xlabel('Train Epoch(K)', font = font1)
    if CPI_mode:
        plt.ylabel('CPI Error Percentage', font = font1)
        #plt.gca().set(xlabel = , ylabel = )
    else:
        plt.ylabel('Power Error Percentage', font = font1)
        #plt.gca().set(xlabel = 'Train Epoch(K)', ylabel = 'Power Error Percentage')
    #plt.title("CPI Model: train and eval loss ", fontsize=15)
    fig_legend = plt.legend(fontsize = 10, loc = 'upper right')
    #set(fig_legend,'Fontname', 'Times New Roman','FontWeight','bold','FontSize',20)

    plt.savefig(fig_name, dpi=dpi_set)
    plt.show()

def to_percent(temp, position):
  return '%1.0f'%(100*temp) # + '%'

def read_cdf_file(CPI_mode, opt):
    x = []
    y = []
    sample_num = 116
    if CPI_mode:
        CPI_mode_str = 'CPI'
    else:
        CPI_mode_str = 'power'
    exceed_flag = False
    cdf_file = open('log/' + CPI_mode_str + '-' + str(sample_num) + opt + '-cdf.txt', 'r')
    for str_line in cdf_file:
        if exceed_flag:
            break
        str_line_array = str_line.split(' ')
        error_cdf = float(str_line_array[2])
        if 1 < error_cdf:
            exceed_flag = True
        #index_array.append(int(str_line_array[0]))        
        x.append(float(str_line_array[1]))
        y.append(error_cdf)
    cdf_file.close()
    return [x, y]

def plt_cdf(X_1_print, Y_1_print, X_2_print, Y_2_print, fig_name, CPI_mode):
    [X_1_print, Y_1_print] = read_cdf_file(CPI_mode, opt = '_')
    [X_2_print, Y_2_print] = read_cdf_file(CPI_mode, opt = '_opt')

    fig_name = 'fig/loss_'
    if bench_indivisual_model:
        fig_name += bench_indivisual_model + '_' + 'sample-id-' + str(sample_id)
    
    #plt.subplot(121+(CPI_mode-1))
    plt.scatter(X_1_print, Y_1_print, c = base_color, s=1)
    plt.scatter(X_2_print, Y_2_print, c = iDSE_color, s=1)
    plt.plot(X_1_print, Y_1_print, color=base_color, marker='.', markersize=1, linestyle = '--', label = 'random')
    plt.plot(X_2_print, Y_2_print, color=iDSE_color, marker='.', markersize=1, label = 'iDSE')
    if CPI_mode:
        mode_str = 'CPI'
        fig_name = fig_name + 'cpi'
    else:
        mode_str = 'Power'
        fig_name = fig_name + 'core-power'        
    fig_name += '-cdf.pdf'
    font1 = {'size' : 12}
    plt.xlabel(mode_str + ' Error Percentage', font = font1)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))    
    plt.ylabel('Percentage of Design Points', font = font1)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))    
    #plt.title("CPI Model: train and eval loss ", fontsize=15)
    plt.legend(fontsize = 14, loc = 'lower right')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(linestyle = '--')
    plt.savefig(fig_name,  dpi=dpi_set)
    plt.show()


sample_sum = 2300

def plot_error_fig():
    #write_xls()
    all_data = pd.read_excel('error_sample/' + 'error.xlsx')
    for CPI_mode in range(2):
        if CPI_mode:
            fig_name = 'CPI'
        else:
            fig_name = 'Power'
        fig_name_str = fig_name + '-error.pdf'
        #seaborn.set_style('darkgrid')
        #seaborn.palplot(seaborn.cubehelix_palette(4, start = 0, rot = 0, dark = 0, light = .95, reverse = True))
        #seaborn.palplot(seaborn.light_palette('black')) #按照green做浅色调色盘
        #flatui = ["white", "white", "black", "black", "#34495e", "#2ecc71"]
        #sns.palplot(sns.color_palette(flatui))
        sns.set_theme(style="whitegrid")
        ax = sns.violinplot(x = "sample"
                        , y = fig_name
                        , hue="method"
                        #, palette="Set2"
                        , palette = {"random": "white", "iDSE": "gray"}
                        #, order = {"baseline", "iDSE"}
                        , data = all_data
                        #, split=True
                        , scale = 'count'
                        , color = 'red'
                        #, palette = 'deep'
                       )
        font1 = {'size' : 15}
        ax.set_xlabel('Sampling Nums', font = font1)
        ax.set_ylabel(fig_name + ' Error Distribution', font = font1)
        plt.savefig(fig_name_str,  dpi=dpi_set)
        plt.show()
    exit(1)

def read_time():
    simulation_time = 130 / 60
    base_time_list = []
    opt_time_list = []
    time_file = open('log/time.log', 'r')
    time_value_accumulate = 0
    for time_file_line in time_file:
        str_split = time_file_line.split(' ')
        str_sample = int(str_split[0])
        str_time = str_split[1].split(':')
        time_value = (float(str_time[1])+ float(str_time[2]) / 60) * 0.4
        time_value_accumulate += time_value
        base_time_list.append(time_value + str_sample * simulation_time)
        opt_time_list.append(time_value_accumulate + str_sample * simulation_time)
    time_file.close()
    return [base_time_list, opt_time_list]

def fit_func(x, a, b):
    return a * np.power(x, b)

def plot_loss_line(CPI_mode):
    if CPI_mode:
        str_mode = 'cpi'
        str_mode_2 = 'CPI'
    else:
        str_mode = 'power'
        str_mode_2 = 'Power'
    #str_mode = 'power'
    train_size_list = []
    epoch_list = []
    loss_list = []
    opt_train_size_list = []
    opt_epoch_list = []
    opt_loss_list = []
    head_flag = True
    sample_log_file_ = open('log/' + str_mode + '_sample_log_file.txt', 'r')
    for sample_data_str in sample_log_file_:
        if head_flag:
            head_flag = False
            continue
        sample_data_str_split = sample_data_str.split(' ')
        if 5 < len(sample_data_str_split):
            train_size = int(sample_data_str_split[0])/sample_sum
            loss = 1- float(sample_data_str_split[2])
            opt_train_size = int(sample_data_str_split[3])/sample_sum
            opt_loss = 1 - float(sample_data_str_split[5].strip('\n'))
            train_size_list.append(train_size)
            loss_list.append(loss)
            opt_train_size_list.append(opt_train_size)
            opt_loss_list.append(opt_loss)
    sample_log_file_.close()

    if 1:
        popt, pcov = curve_fit(fit_func, train_size_list, loss_list)
        #print(popt)
        loss_list_fit = fit_func(train_size_list, *popt)
        fit_coler = 'red'
        #plt.scatter(train_size_list, loss_list_fit, c = fit_coler, s=7)
        fit_func_str = 'y=' + str(round(popt[0], 2)) + 'x^' + str(round(popt[1], 2))
        print('CPI_mode=' + str(CPI_mode) + ' random curve: ' + fit_func_str)
        #plt.plot(train_size_list, loss_list_fit, color = fit_coler, marker='.', markersize=7, linestyle = '-', label = 'random ' + fit_func_str)

    [base_time_list,opt_time_list] = read_time()

    #plt.subplot(121+(CPI_mode-1))
    plt.scatter(base_time_list, loss_list, c = base_color, s=7)
    plt.scatter(opt_time_list, opt_loss_list, c = iDSE_color, s=7)
    plt.plot(base_time_list, loss_list, color=base_color, marker='.', markersize=7, linestyle = '--', label = 'random')
    plt.plot(opt_time_list, opt_loss_list, color=iDSE_color, marker='+', markersize=7, label = 'iDSE')
    font1 = {'size' : 18}
    #2303
    plt.xlabel('Time(minute)', font = font1)
    #plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.ylabel(str_mode_2 + ' Accuracy(%)', font = font1)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    #plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    #plt.title("CPI Model: train and eval loss ", fontsize=15)
    plt.legend(fontsize = 17, loc = 'lower right')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(linestyle = '--')
    plt.savefig(bench_indivisual_model + '_' + str_mode_2 + '_loss_line.pdf',  dpi=dpi_set)
    plt.show()

def plot_dsp_coverage(area_mode):
    if area_mode:
        str_mode = 'area'
        str_mode_2 = 'CPIxArea'
    else:
        str_mode = 'power'
        str_mode_2 = 'CPIxPower'
    found_pareto_num_list = [0]
    sample_num_list = [0]
    found_pareto_num_base_list = [0]
    sample_num_base_list = [0]
    head_flag = True
    sample_log_file_ = open('log/' + str_mode + '-hit_file_choose.txt', 'r')
    for sample_data_str in sample_log_file_:
        if head_flag:
            head_flag = False
            continue
        sample_data_str_split = sample_data_str.split(' ')
        if 5 < len(sample_data_str_split):        
            if 'choose_one_in_pareto' in sample_data_str:
                all_pareto_num = int(sample_data_str_split[6].strip('\n'))
                found_pareto_num = int(sample_data_str_split[2])/all_pareto_num
                sample_num = int(sample_data_str_split[4])/sample_sum
                #all_pareto_num_.append(train_size)
                found_pareto_num_list.append(found_pareto_num)
                sample_num_list.append(sample_num)
            else:
                #all_pareto_num = int(sample_data_str_split[6].strip('\n'))
                found_pareto_num_base = int(sample_data_str_split[2])/all_pareto_num
                sample_num_base = int(sample_data_str_split[4])/sample_sum
                #all_pareto_num_.append(train_size)
                found_pareto_num_base_list.append(found_pareto_num_base)
                sample_num_base_list.append(sample_num_base)
    sample_log_file_.close()

    #plt.subplot(121+(CPI_mode-1))
    our_color = 'black'
    max_coverage = max(found_pareto_num_list)
    plt.scatter(sample_num_list, found_pareto_num_list, c = our_color, s=7)
    plt.plot(sample_num_list, found_pareto_num_list, color=our_color, marker='*', markersize=7, label = 'iDSE')

    plt.scatter(sample_num_base_list, found_pareto_num_base_list, c = base_color, s=7)
    plt.plot(sample_num_base_list, found_pareto_num_base_list, color=base_color, marker='+', markersize=7, label = 'random')

    #plt.scatter([0,max_coverage], [0,max_coverage], c = base_color, s=7)
    #plt.plot([0,max_coverage], [0,max_coverage], color=base_color, marker='+', markersize=7, linestyle = '--', label = 'baseline')

    font1 = {'size' : 16}
    #2303
    plt.xlabel('Sample Ratio (%)', font = font1)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.ylabel('Coverage (%)', font = font1)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    #plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    #plt.title("CPI Model: train and eval loss ", fontsize=15)
    plt.legend(fontsize = 17, loc = 'upper left')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(linestyle = '--')
    plt.savefig(bench_indivisual_model + '_' + str_mode_2 + '_coverage.pdf',  dpi=dpi_set)
    plt.show()

if __name__ == '__main__':
    #sample_error_distribute(models = [], CPI_mode = 1, sample_number = 2)
    #plot_error_fig()
    if 0:
        plot_pareto_optimality(None, None, None)
        exit(0)
    if 0:
        plot_pareto_optimality_all()
        exit(0)        
    if 0:
        plot_loss_line(CPI_mode = 1)
        plot_loss_line(CPI_mode = 0)
        exit(0)
    if 0:
        plt_cdf(None, None, None, None, None, CPI_mode = 1)
        plt_cdf(None, None, None, None, None, CPI_mode = 0)
        exit(0)
    if 1:
        plot_dsp_coverage(area_mode = area_mode)
        #plot_dsp_coverage(area_mode = 1)
        exit(0)
    try:
        main()
        plot_pareto_optimality(None, None, None)
        if 1:
            plt_cdf(None, None, None, None, None, CPI_mode = 1)
            plt_cdf(None, None, None, None, None, CPI_mode = 0)
            #exit(0)
        #if 0 == demo:
        #    plot_pareto_optimality_all()
    except:
        print('main except')
    log_file.close()