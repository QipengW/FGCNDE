# State Feedback Enhanced Graph Differential Equation for Spatio-temporal Time Series Prediction
# Dataloader
The three zip files store the three data sets used in this article and the code for loading the data sets. The code has been divided into training set, verification set and test set and can be run directly. Among them, the PEMS04 data set failed to be successfully uploaded due to upload size restrictions. Please feel free to contact us if necessary.
# Model
The forward propagation process of the algorithm proposed in this article is defined in the ST-GDE.py file. ST-GDE accepts [B, N, H] dimension input and outputs [B, N, T], where B is batch_size, N is the number of nodes, H is the historical input length, and T is the length to be predicted. In this article H=5, T=10.
# Requirement
torch==1.10.2
numpy==1.19.2
pandas==1.1.5
scikit-learn==0.24.2
