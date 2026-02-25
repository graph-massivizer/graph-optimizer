from . import performance_model
import models.utils as utils

def predict(hardware):
    performance = performance_model.predict(hardware)
    wattage = hardware['cpus']['wattage']

    return f'({performance}) / 1000000000000 * {wattage}m'