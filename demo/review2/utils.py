import matplotlib.pyplot as plt

BILLION = 1000000000
MILLION = 1000000

def plot_runtimes(bars):
    keys = list(bars.keys())
    runtimes = [bars[k][0]/BILLION for k in keys]
    energies = [bars[k][1]/MILLION for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Runtime subplot
    ax1.bar(keys, runtimes)
    ax1.set_ylabel("Runtime (s)")
    ax1.set_title("Runtime")

    # Energy subplot
    ax2.bar(keys, energies)
    ax2.set_ylabel("Energy (MJ)")
    ax2.set_title("Energy")

    plt.show()

def convert_output(output):
    output = output.strip().split('\n')[-1].split(' ')

    return (float(output[3])*BILLION, float(output[6])*MILLION)

def get_error(bars):
    prediction = bars['Prediction']
    validation = bars['Validation']

    runtime_error = abs(validation[0]-prediction[0])/validation[0]
    energy_error = abs(validation[1]-prediction[1])/validation[1]

    return runtime_error, energy_error