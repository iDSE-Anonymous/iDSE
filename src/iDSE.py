import torch
import numpy as np
from sklearn import preprocessing
import copy
import os
import random
import time
import math
from datetime import datetime

import global_setting
global_setting._init()
area_mode = global_setting.get_global_var('area_mode')
bench_indivisual_model = global_setting.get_global_var('bench_indivisual_model')
simpoint_id = global_setting.get_global_var('simpoint_id')
sample_num = global_setting.get_global_var('sample_num')
sample_id = global_setting.get_global_var('sample_id')
BENCH_SIMPOINT_INDEX = global_setting.get_global_var('BENCH_SIMPOINT_INDEX')
CASE_VERSION_INDEX = global_setting.get_global_var('CASE_VERSION_INDEX')

import read_input
from read_input import *
import key_point_generation
from key_point_generation import *
import MLP_model
from MLP_model import *
import figure_plot
from figure_plot import *

log_name = ''
if bench_indivisual_model:
    log_name += bench_indivisual_model + '-' + str(1)
log_name += '-' + str(sample_id) + '-' + str(area_mode)
log_name += '-train_eval.log'
log_file = open(log_name, 'w')

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

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def main():

    #setup_seed(4)

    startTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    startTime = datetime.strptime(startTime,"%Y-%m-%d %H:%M:%S")

    max_epoch_number = 20000
    print_peroid = 200
    loss_delta_thredhold = 0.0000000001
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

    [raw_data, final_cpi_labels, final_power_labels, real_y_label_array, bench_info, batch_number, input_length] = data_loader('./data_all_simpoint/')

    [all_input, min_max_scaler] = data_preprocess(raw_data)
    #real_y_label_array: cpi, bench_id, version
    [all_input, final_cpi_labels, final_power_labels, real_y_label_array] = shuffle(all_input, final_cpi_labels, final_power_labels, real_y_label_array)
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
            #if choose_from_dsp and 1 < sample_id:
            #    train_data_size_opt_limit = 1000 #sample_id * 25

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
    if cdf_enable:
        print('cdf output --------------------------------------------------------------------------------------')
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
        print('train_loss_fig --------------------------------------------------------------------------------------')
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
    if 0:
        plot_dsp_coverage(area_mode = area_mode)
        exit(0)
    try:
        main()
    except:
        print('main except')
    log_file.close()