import heapq
import torch
import numpy as np

import global_setting
area_mode = global_setting.get_global_var('area_mode')
bench_indivisual_model = global_setting.get_global_var('bench_indivisual_model')
simpoint_id = global_setting.get_global_var('simpoint_id')
sample_num = global_setting.get_global_var('sample_num')
sample_id = global_setting.get_global_var('sample_id')
BENCH_ID_INDEX = global_setting.get_global_var('BENCH_ID_INDEX')
BENCH_SIMPOINT_INDEX = global_setting.get_global_var('BENCH_SIMPOINT_INDEX')
CASE_VERSION_INDEX = global_setting.get_global_var('CASE_VERSION_INDEX')
inst_radio_mode = global_setting.get_global_var('inst_radio_mode')

gen_choose_list_flag = 0
choose_from_dsp = 1

import read_input
from read_input import *
import figure_plot
from figure_plot import write_error

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

    demo = 0
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
        #log_file.write(filename + ': pareto_points_hit_num/all = ' + str(pareto_points_hit_num) + ' / ' + str(len(points_config)) + ' \n')
        hit_file = open('log/hit.log', 'a')
        hit_file.write(filename + ' ' + str(sample_num) + ' : pareto_points_hit_num/all = ' + str(pareto_points_hit_num) + ' / ' + str(len(points_config)) + ' \n')
        hit_file.close()
    return [points_x, points_y, points_x_pareto, points_y_pareto, pareto_configs]