# **Gas Turbine Electricity Prediction with LSTM Neural Networks**

## **Overview**

This repository hosts a project focused on predicting the electricity output of gas turbines using **Long Short-Term Memory (LSTM)** neural networks. The workflow spans **data preprocessing**, **feature engineering**, **model development**, **training**, **evaluation**, and **deployment-ready model saving**. Leveraging LSTMs' ability to handle sequential data, the project achieves competitive metrics, showcasing their effectiveness in time-series prediction tasks.

----------

## **Project Description**

Efficiently predicting gas turbine performance is critical for optimizing energy generation. This project employs **deep learning techniques** to model the temporal relationship between input control signals and power output. By analyzing past readings, the model aims to enhance prediction accuracy, reduce reliance on manual monitoring, and lower operational costs.


### **Objectives:**

-   Build and train an LSTM-based model for gas turbine electricity prediction.
-   Optimize hyperparameters to enhance prediction accuracy.
-   Evaluate the model using key metrics: **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**, and **Root Mean Squared Error (RMSE)**.
-   Achieve an RMSE of **less than 400**.
-   Eliminate the reliance on traditional methods for predicting electricity power output, which require extensive tools, measurements, and human monitoring, thereby reducing maintenance costs and complexity.

----------

## **Dataset**

The dataset comprises operational measurements from a gas turbine system, organized as time series data. [Source](https://archive.ics.uci.edu/dataset/994/micro+gas+turbine+electrical+energy+prediction)

**Features:**

1.  **Time**: Sequential time stamps.
2.  **Input Voltage**: Control signal for turbine operation.
3.  **Power Output**: Generated electrical power.

### **Key Characteristics:**

-   Contains **eight distinct time series**, with durations ranging from **1.8 to 3.3 hours** (~6,495–11,820 data points per series).
-   Time series represent two scenarios:
    -   **Rectangular Signals:** Sudden input voltage changes, with delayed power output.
    -   **Continuous Signals:** Gradual input voltage changes, with minimal delays.

### **Data Splits:**

-   **Training Data:** 42,349 samples.
-   **Validation Data:** 10,588 samples.
-   **Test Data:** 18,282 samples.

### **Recommended Splits:**

-   Training: Experiments 1, 9, 20, 21, 23, and 24.
-   Testing: Experiments 4 and 22.  
    **Performance Metric:** Root Mean Squared Error (RMSE).
```
├── data/                  # Train & Test datasets
├── model/                 # Saved models
├── plots/                 # Visualizations
├── scripts/               # Core scripts for the project
│   ├── load_data.py       # Data loading and preprocessing, feature engineering
│   ├── model.py           # LSTM model architecture
│   ├── plot_metrics.py    # Training history visualization
├── notebooks/             # Jupyter notebooks for EDA and experimentation
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── train_model.py         # Model training script
```

----------

## **Model Architecture**

The model utilizes **TensorFlow/Keras** to implement a sequential LSTM network.  
**Layers:**

1.  **LSTM Layers**: Extract sequential features.
2.  **Dropout Layers**: Prevent overfitting.
3.  **Dense Layers**: Predict output power.

### **Final Architecture:**


| Layer Name          | Output Shape   | Parameters |
|---------------------|----------------|------------|
| LSTM (lstm)         | (None, 5, 32)  | 4,352      |
| Dropout (dropout)   | (None, 5, 32)  | 0          |
| LSTM (lstm_1)       | (None, 32)     | 8,320      |
| Dropout (dropout_1) | (None, 32)     | 0          |
| Dense (dense)       | (None, 16)     | 528        |
| Dense (dense_1)     | (None, 1)      | 17         |


**Total Parameters:** 13,217  
**Trainable Parameters:** 13,217  
**Non-Trainable Parameters:** 0

### **Loss Function:**

-   **Mean Squared Error (MSE)**

----------

## **Results**

### **Performance Metrics:**

-   **Loss:** Gradual convergence, indicating effective training.
-   **RMSE:** Achieved **366.157** on the test set.

### **Key Challenges:**

1.  **Overfitting:** Limited dataset size required dropout layers and regularization.
2.  **Non-linear Dynamics:** Input-output patterns (e.g., delayed ramps) were challenging for traditional methods.
    #### Output
    ![image](https://github.com/user-attachments/assets/740476cc-938a-4fbc-8204-ccb4caacb72b)
    #### Input
    ![image](https://github.com/user-attachments/assets/111d35df-83fb-4c1d-9d64-f23418463b79)
    #### Output
    ![image](https://github.com/user-attachments/assets/e06ab9b6-b5c3-48b6-ba40-e6c5fe6bb104)
    #### Input
    ![image](https://github.com/user-attachments/assets/f8d390f1-94a6-49cd-8c18-916fe35eca4e)
3.  **Comparison with Other Models:**
    -   Linear models and ensemble methods (e.g., Random Forest, XGBoost) failed to generalize well.
    -   LSTM models excelled due to their ability to capture sequential dependencies.

----------

## **Installation and Usage**

### **Prerequisites:**

-   Python 3.8 or higher
-   TensorFlow==2.6.3
-   Pandas, NumPy, Matplotlib, and Scikit-learn

### **Installation Steps:**

1.  Clone the repository:
```
    git clone https://github.com/Jatin-Mehra119/Micro-Gas-Turbine-Electrical-Energy-Prediction-Modeling.git
    cd gas-turbine-prediction
```

2.  Install dependencies:
```
    pip install -r requirements.txt
```
### **Usage:**

To train the model, run:

`python -m train_model.py` 

The trained models will be saved in the `model/` directory.

----------

## **Conclusion**

This project highlights the potential of **LSTM neural networks** for modeling complex time-series behaviors in gas turbines. By addressing challenges such as overfitting and non-linear dynamics, the model demonstrates superior performance compared to traditional approaches.

----------

## **License**

This project is licensed under the **MIT License**. For details, refer to the `LICENSE` file.

For queries or further assistance, contact: **jatinmehra@outlook.in**.


## **References**

-   Bielski, P. & Kottonau, D. (2024). Micro Gas Turbine Electrical Energy Prediction [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C58S4T](https://doi.org/10.24432/C58S4T).
