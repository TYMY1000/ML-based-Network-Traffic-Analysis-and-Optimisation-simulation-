from flask import Flask, render_template, jsonify, request
import torch
import torch.nn as nn
import pickle
import numpy as np
import subprocess
import json
import pandas as pd
import os
import time
import random

app = Flask(__name__)

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# Load model and scaler
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    input_features = ["Jitter"]
    input_dim = len(input_features)

    model = MLP(input_dim, 128, 3)
    model.load_state_dict(torch.load("model/network_traffic_model.pth", map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    print("Model/Scaler load error:", e)
    scaler = None
    model = None

# Counter for simulated jitter delay
capture_counter = 0
qos_applied = False  # Global flag to avoid multiple applications

def get_iperf3_stats():
    global capture_counter, qos_applied
    capture_counter += 1

    try:
        iperf_path = r"C:\iperf\iperf3.exe"
        if not os.path.exists(iperf_path):
            raise FileNotFoundError("iperf3.exe not found")

        cmd = [iperf_path, "-c", "127.0.0.1", "-u", "-p", "5201", "-t", "5", "-J"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

        if result.returncode != 0:
            print("iperf3 error:", result.stderr.strip())
            jitter = 0.0
        else:
            output = json.loads(result.stdout)
            jitter = 0.0
            if "end" in output:
                if "streams" in output["end"] and output["end"]["streams"]:
                    stream = output["end"]["streams"][0]
                    udp_data = stream.get("udp", {})
                    jitter = udp_data.get("jitter_ms", 0.0)

        # Simulate QoS effect: reduce jitter by 50% if optimization is enabled
        if qos_applied:
            jitter *= 0.3
            print(f"[Optimized] Jitter reduced to {jitter:.2f} ms")
        if random.random() < 0.2:
            jitter = random.uniform(10.0, 20.0)
            print(f"Simulated high jitter: {jitter:.2f} ms")
        return {"Jitter": jitter}

    except Exception as e:
        print("Subprocess error:", e)
        return {"Jitter": 0.0}


# QoS Optimization functions (for Windows PowerShell)
def apply_qos_policy():
    try:
        print("[⚙] Applying QoS policy for UDP prioritization...")
        subprocess.run([
            "powershell",
            "-Command",
            "New-NetQosPolicy -Name 'PrioritizeUDP' -AppPathNameMatchCondition 'iperf3.exe' "
            "-IPProtocolMatchCondition UDP -DSCPAction 46"
        ], check=True)
        print("[✅] QoS policy applied.")
    except subprocess.CalledProcessError as e:
        print("[❌] QoS apply error:", e)

def remove_qos_policy():
    try:
        print("[🧹] Removing existing QoS policy...")
        subprocess.run([
            "powershell",
            "-Command",
            "Remove-NetQosPolicy -Name 'PrioritizeUDP' -Confirm:$false"
        ], check=True)
        print("[✅] QoS policy removed.")
    except subprocess.CalledProcessError as e:
        print("[❌] QoS remove error:", e)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET"])
def predict():
    global qos_applied

    if not model or not scaler:
        return jsonify({"error": "Model or scaler not loaded properly"}), 500

    data = get_iperf3_stats()
    print("Live data:", data)

    try:
        df = pd.DataFrame([data])
        scaled = scaler.transform(df[["Jitter"]].values)
        tensor_input = torch.tensor(scaled, dtype=torch.float32)

        with torch.no_grad():
            prediction = model(tensor_input)
            predicted_class = prediction.argmax(dim=1).item()

        status = ["NOT congested", "Congested", "HIGHLY congested"]
        predicted_status = status[predicted_class]

        # QoS logic based on prediction
        if predicted_class == 2:  # HIGHLY congested
            if not qos_applied:
                apply_qos_policy()
                qos_applied = True
        else:
            if qos_applied:
                remove_qos_policy()
                qos_applied = False

        return jsonify({
            "input": data,
            "prediction": predicted_status,
            "optimization": "Applied" if qos_applied else "Not applied"
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500

@app.route("/optimize", methods=["POST"])
def toggle_optimization():
    global qos_applied
    try:
        if not qos_applied:
            # Apply QoS policy
            apply_qos_policy()
            qos_applied = True
            return jsonify({"active": True, "status": "QoS applied"})
        else:
            # Remove QoS policy
            remove_qos_policy()
            qos_applied = False
            return jsonify({"active": False, "status": "QoS removed"})
    except Exception as e:
        return jsonify({"error": f"Failed to toggle QoS: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
