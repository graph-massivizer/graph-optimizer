from flask import Flask, request, jsonify
import json
import subprocess
import pathlib
from datetime import datetime

prediction_script = str(pathlib.Path(__file__).parent.parent.absolute()) + '/models/prediction.py'
print(prediction_script)
app = Flask(__name__)

def predict(request, evaluate):
    hardware = request.form.get('hardware')
    bgo_dag = request.form.get('bgo_dag')

    if (evaluate):
        graph_props = request.form.get('graph_props')
        output = subprocess.check_output(['python3', prediction_script, hardware, bgo_dag, graph_props])
        return output

    output = subprocess.check_output(['python3', prediction_script, hardware, bgo_dag])
    return output

def greenify_output(output):
    greenifier_output = {"tasks": json.loads(output)}

    for bgo in greenifier_output['tasks']:
        bgo['runTimes'] = {perf['host']: perf['runtime'] for perf in bgo['performances']}
        bgo['energyConsumption'] = {perf['host']: perf['energy'] for perf in bgo['performances']}
        bgo['submissionTime'] = datetime.today().strftime('%Y-%m-%d')
        # TODO: These values are hardcoded for now
        bgo['cpuUsage'] = 10000
        bgo['memCapacity'] = 1000000
        bgo['cpuCount'] = 1

        del bgo['performances']

    return greenifier_output


@app.route('/models', methods=['POST'])
def models():
    return predict(request, False)


@app.route('/evaluate', methods=['POST'])
def evaluate():
    return predict(request, True)

@app.route('/greenifier', methods=['POST'])
def greenifier():
    output = predict(request, True)
    return greenify_output(output)
