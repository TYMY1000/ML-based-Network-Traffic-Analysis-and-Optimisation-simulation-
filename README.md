# ML-based-Network-Traffic-Analysis-and-Optimisation-simulation-


---

#  Network Traffic Classification & Optimization using MAML + MLP

This project applies **Model-Agnostic Meta-Learning (MAML)** combined with a **Multi-Layer Perceptron (MLP)** to classify network traffic using jitter values as input. It is designed to simulate adaptive learning for different traffic types, even in the presence of data imbalance. The system is capable of quickly generalizing to new conditions and can serve as the foundation for QoS-based traffic optimization or SDN (Software Defined Networking) strategies.

##  Overview

* **Input Feature**: Jitter
* **Target Output**: Traffic Classes (e.g., 0 = Real-time traffic, 1 = Bulk transfer, 2 = Normal browsing)
* **Core Model**: PyTorch-based MLP
* **Training Strategy**: MAML with stratified sampling
* **Goal**: Improve traffic classification and prepare for integration into real-time optimization tools

##  Features

*  Adaptive MAML loop for meta-learning
*  Class-weighted loss & stratified sampling to handle class imbalance
*  Evaluation metrics include accuracy, precision, recall, F1-score, and ROC AUC
*  Visualizations: loss curves, confusion matrices, and metric comparison charts
*  Prepared for further expansion into intelligent traffic control systems




##  How to Run

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Prepare your dataset in CSV 

3. Run the training script:

   ```
   python train_model.py
   ```

4. Outputs such as trained model, plots, and evaluation reports will be saved in the `model/` directory.

##  Future Improvements

* Integrate real-time traffic control logic using classification results
* Connect with SDN controllers (e.g., Ryu) for live optimization
* Add features like bandwidth, latency, packet loss, etc.
* Explore advanced loss functions like focal loss or contrastive loss

## Author

**Awomolo TRotimi**
Feel free to connect or contribute!


