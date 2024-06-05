from flask import Flask, request, jsonify
import json
import subprocess

app = Flask(__name__)

def predict(request, evaluate):
    hardware = request.form.get('hardware')
    bgo_dag = request.form.get('bgo_dag')

    if (evaluate):
        graph_props = request.form.get('graph_props')
        output = subprocess.check_output(['python3', '../models/prediction.py', hardware, bgo_dag, graph_props])
        return output

    output = subprocess.check_output(['python3', '../models/prediction.py', hardware, bgo_dag])
    return output


@app.route('/models', methods=['POST'])
def models():
    return predict(request, False)


@app.route('/evaluate', methods=['POST'])
def evaluate():
    return predict(request, True)
