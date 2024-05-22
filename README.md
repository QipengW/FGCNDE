# FGCNDE
# dataloader
The three zip files store the three data sets used in this article and the code for loading the data sets. The code has been divided into training set, verification set and test set and can be run directly. Among them, the PEMS04 data set failed to be successfully uploaded due to upload size restrictions. Please feel free to contact us if necessary.
# model
The forward propagation process of the algorithm proposed in this article is defined in the FGCNDE.py file. FGCNDE accepts [B, N, H] dimension input and outputs [B, N, T], where B is batch_size, N is the number of nodes, H is the historical input length, and T is the length to be predicted. In this article H=5, T=10.
# Reproduce article results
We saved the parameters of the FGCNDE model after running on the three data sets in the Weights trained on three data sets folder, and we can directly use our trained parameters to display the results in our article.
