import torch
import torch.nn as nn
import torch.nn.functional as F

import global_setting
area_mode = global_setting.get_global_var('area_mode')
bench_indivisual_model = global_setting.get_global_var('bench_indivisual_model')
simpoint_id = global_setting.get_global_var('simpoint_id')
sample_num = global_setting.get_global_var('sample_num')
sample_id = global_setting.get_global_var('sample_id')
BENCH_ID_INDEX = global_setting.get_global_var('BENCH_ID_INDEX')
BENCH_SIMPOINT_INDEX = global_setting.get_global_var('BENCH_SIMPOINT_INDEX')
CASE_VERSION_INDEX = global_setting.get_global_var('CASE_VERSION_INDEX')

import read_input
from read_input import *

class Loss_Fun(nn.Module):
    def __init__(self):
        super(Loss_Fun, self).__init__()
        
    def forward(self, y, label):
        return torch.abs((y / label) - 1)

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
        #log_file.write(CPI_mode_str + ' 5\% coverage error_limit=' + str(error_limit) + '\n')
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
        #log_file.write(CPI_mode_str + ' 5\% coverage error_limit=' + str(error_limit) + '\n')
        return error_cdf