# Fraud Detection with SOMs and Deep Learning

This project focuses on detecting fraudulent credit card applications using Self-Organizing Maps (SOMs) and combining them with Artificial Neural Networks (ANNs) for a hybrid model.

---

## In This Project, You Will Learn:

1. The intuition behind Self-Organizing Maps (SOMs) and their use in unsupervised learning.
2. How to transition from unsupervised learning to supervised learning using deep learning techniques.
3. How to visualize fraud detection results and evaluate predictions.

---

## **Self-Organizing Maps (SOMs)**

### **Intuition of SOMs**

SOMs are unsupervised neural networks that project high-dimensional data into a lower-dimensional space, preserving the structure and identifying clusters.

### **Steps in `code2.py`**

1. **Feature Scaling**: 
   - Applies Min-Max Scaling to normalize the dataset between 0 and 1.

2. **Training the SOM**:
   - A 10x10 SOM is trained to identify clusters in the data.
   - SOM identifies outliers (potential frauds) by visualizing the distance map.

3. **Visualization**:
   - A heatmap is plotted to show SOM clusters and highlight fraud-prone areas.

4. **Fraud Detection**:
   - The SOM's winning nodes are used to extract the fraud entries.

---

## **From Unsupervised to Supervised Learning**

### **Overview of Hybrid Approach**

Combining the strengths of SOMs and ANNs, the hybrid model uses SOM to label potential frauds and ANN to predict fraud probabilities.

---

## **Deep Learning with ANN**

### **Steps in `mega_case_study.py`**

#### **1. Preprocessing**:
- Fraud labels are assigned based on SOM outputs.
- The features (`customers`) are scaled using Standard Scaling.

#### **2. ANN Construction**:
- The ANN has:
  - **Input Layer**: Accepts 15 features.
  - **Hidden Layer**: Uses ReLU activation for non-linearity.
  - **Output Layer**: Uses Sigmoid activation for predicting probabilities.

#### **3. Model Compilation and Training**:
- Optimized using the Adam optimizer and binary cross-entropy loss function.
- Trained on fraud-labeled data for 2 epochs.

### **Activation Functions**
![Activation Functions](Images/activation_functions.png)

#### Sigmoid Function:
- Outputs probabilities between 0 and 1, suitable for binary classification tasks.

---

## **Visualization**

### **SOM Visualization**

![SOM Heatmap](Images/som_heatmap.png)

- Red areas represent clusters with higher potential fraud risk.

### **ANN Predictions**

![ANN Predictions](Images/ann_predictions.png)

- Fraud probabilities for each customer are displayed in descending order for better evaluation.

---

## **How to Use**

### Prerequisites
1. Install Python 3.x and required libraries:
   ```bash
   pip install numpy pandas matplotlib keras minisom
   ```
2. Place the dataset (`Credit_Card_Applications.csv`) in the project directory.

### Execution
1. Run `code2.py` to train the SOM and visualize fraud clusters.
2. Run `mega_case_study.py` to train the ANN and predict fraud probabilities.

---

## **Outputs**

1. **Fraud List**:
   - Extracted from SOM mappings.

2. **SOM Heatmap**:
   - Clustering visualization.

3. **Fraud Probabilities**:
   - ANN outputs for each customer.

---

## **Contributions and Acknowledgments**

- SOM implementation uses the `minisom` library.
- ANN framework is built using `Keras` with `TensorFlow` backend.

Feel free to enhance the scripts and contribute to the project!