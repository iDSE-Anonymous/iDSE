import os
import numpy as np
import copy
import math
from sklearn import preprocessing

import global_setting
area_mode = global_setting.get_global_var('area_mode')
bench_indivisual_model = global_setting.get_global_var('bench_indivisual_model')
simpoint_id = global_setting.get_global_var('simpoint_id')
sample_num = global_setting.get_global_var('sample_num')
sample_id = global_setting.get_global_var('sample_id')
BENCH_SIMPOINT_INDEX = global_setting.get_global_var('BENCH_SIMPOINT_INDEX')
CASE_VERSION_INDEX = global_setting.get_global_var('CASE_VERSION_INDEX')
inst_radio_mode = global_setting.get_global_var('inst_radio_mode')

input_length_max = 0
simpoint_mode = 1

CASE_MAX_NUM = 52
#416.1/2/3 is ignored due to compiler error

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


def read_config(config_dir):
    configs = []
    #config_list = [config for config in ]

    #config_dir_name = config_dir
    for config_dir_name in os.listdir(config_dir):
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

def data_loader(data_path):
    configs = configs_all
    case_length = len(configs[0]['params'])
    #print(configs[0]['params'])

    if bench_indivisual_model is None or simpoint_mode:
        bench_soft_info_length = 2 * len(specific_version)  #1 for bench tag(one hot coding)
        if inst_radio_mode:
            bench_soft_info_length += 5 #inst_radio length
        print('bench_soft_info_length = ' + str(bench_soft_info_length))
        #log_file.write('bench_soft_info_length = ' + str(bench_soft_info_length) + '\n')
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
    #log_file.write('case_length = ' + str(case_length) + '\n')            
    #case_length_real  = case_length - len(input_mask_array)
    input_enable = 0
    if input_enable:
        case_length_real = len(input_enable_array)
    else:
        case_length_real = case_length - len(input_mask_array)
        #case_length_real = case_length
    #case_length_real = len(bench_cases[0])
    print('case_length_real = ' + str(case_length_real))
    #log_file.write('case_length_real = ' + str(case_length_real) + '\n')

    print('data cases num all = ' + str(case_num_all))
    #log_file.write('data cases num all = ' + str(case_num_all) + '\n')
    assert(0 < case_num_all)
    case_num = len(final_data)
    print('data cases num = ' + str(case_num))
    #log_file.write('data cases num = ' + str(case_num) + '\n')
    assert(0 < case_num)
    output_bench_info()
    input_length = case_length_real + bench_soft_info_length
    #final_data = np.array(final_data)
    #final_data = torch.Tensor(final_data)
    return [final_data, final_cpi_labels, final_power_labels, final_labels, bench_info, case_num, input_length]


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
    #   print("bench_id: " + str(bench_id))
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
        #filter uselesscase
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