import subprocess 
from benchmarks.microbenchmarks import all_benchmarks

def extract_hardware_characteristics(name, power_draw, include_gpu=False):
    hardware = {
        "name": name,
        "cpus": {
            "wattage": power_draw,
        }
    }

    lscpu = {k.strip(): v.strip() for item in subprocess.check_output(['lscpu']).decode('utf-8').split('\n') for k, _, v in [item.partition(':')]}
    hardware['cpus']['name'] = lscpu['Model name']
    hardware['cpus']['cores'] = int(lscpu['Core(s) per socket'])
    hardware['cpus']['threads'] = int(lscpu['Thread(s) per core']) * hardware['cpus']['cores']
    hardware['cpus']['amount'] = int(lscpu['Socket(s)'])
    hardware['cpus']['clock_speed'] = float(lscpu['CPU max MHz'])
    
    if (include_gpu):
        query_fields = ['name', 'memory.total', 'power.default_limit', 'compute_cap', 'driver_version']
        nvidia_smi = dict(zip(*map(lambda x: [y.strip() for y in x], map(lambda x: x.split(','), subprocess.check_output(['nvidia-smi', f'--query-gpu={",".join(query_fields)}', '--format=csv']).decode('utf-8').strip().split('\n')))))

        hardware['gpus'] = {}
        hardware['gpus']['name'] = nvidia_smi['name']
        hardware['gpus']['memory'] = nvidia_smi['memory.total [MiB]']
        hardware['gpus']['wattage'] = nvidia_smi['power.default_limit [W]']
        hardware['gpus']['compute_capability'] = nvidia_smi['compute_cap']
        hardware['gpus']['driver_version'] = nvidia_smi['driver_version']
    
    return hardware

def hardware_and_microbenchmarks(name, power_draw, include_gpu=False):
    hardware = extract_hardware_characteristics(name, power_draw, include_gpu)
    hardware['cpus']['benchmarks'] = all_benchmarks()
    
    return hardware

if __name__=='__main__':
    name = 'H01'
    power_draw = 35
    include_gpu = True
    print(extract_hardware_characteristics(name, power_draw, include_gpu))
    # print(hardware_and_microbenchmarks(name, power_draw))
