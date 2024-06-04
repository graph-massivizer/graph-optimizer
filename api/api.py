from flask import Flask, request, jsonify
import json
import subprocess

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    hardware = request.form.get('hardware')
    bgo_dag = request.form.get('bgo_dag')
    graph_props = request.form.get('graph_props')

    print(hardware)
    print(bgo_dag)
    print(graph_props)

    output = subprocess.check_output(['python3', '../models/prediction.py', hardware, bgo_dag, graph_props])

    return output