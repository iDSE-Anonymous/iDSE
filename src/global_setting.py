
def _init():
	global area_mode
	area_mode = 0

	global bench_indivisual_model
	#bench_indivisual_model = None
	#bench_indivisual_model = '400.1'
	#bench_indivisual_model = '483.1'
	bench_indivisual_model = '403.2'
	#bench_indivisual_model = '473.1'

	global simpoint_id
	simpoint_id = 1

	global sample_num
	sample_num = 1 # will be set later

	global sample_id
	sample_id = 1

	global sample_id_max
	sample_id_max = sample_id

	global BENCH_ID_INDEX
	BENCH_ID_INDEX = 0

	global BENCH_SIMPOINT_INDEX
	BENCH_SIMPOINT_INDEX = BENCH_ID_INDEX + 1
 	
	global CASE_VERSION_INDEX
	CASE_VERSION_INDEX = BENCH_SIMPOINT_INDEX + 1

	global inst_radio_mode
	inst_radio_mode = 1

def get_global_var(str_var):
	if 'area_mode' == str_var:
		return area_mode
	elif 'bench_indivisual_model' == str_var:
		return bench_indivisual_model
	elif 'simpoint_id' == str_var:
		return simpoint_id
	elif 'sample_num' == str_var:
		return sample_num
	elif 'sample_id' == str_var:
		return sample_id
	elif 'BENCH_ID_INDEX' == str_var:
		return BENCH_ID_INDEX
	elif 'BENCH_SIMPOINT_INDEX' == str_var:
		return BENCH_SIMPOINT_INDEX
	elif 'CASE_VERSION_INDEX' == str_var:
		return CASE_VERSION_INDEX
	elif 'inst_radio_mode' == str_var:
		return inst_radio_mode
	else:
		print('no define global ', str_var)