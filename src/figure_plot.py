import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.size'] = '10'
plt.rcParams['font.weight'] = 'bold'

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