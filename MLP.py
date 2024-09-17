import torch
import numpy as np
from FullyConnectedLayer import FullyConnectedLayer



class MLP:

    def __init__(self, pretrained, **kwargs):

        self.learn_rate = 0.01
        self.batch_size = 1
        self.reduction = "mean"

        if not pretrained:
            input_feature_count = kwargs.get("input_feature_count")
            self.layers = MLP.buildLayers(kwargs.get("mlp_model_config"), input_feature_count)
            # print(f"input_feature_count = {input_feature_count}")
        else:
            self.layers = MLP.loadLayers(kwargs.get("mlp_model_params"))

        self.num_layers = len(self.layers)








    # def inference(self, data):
    #     return self.forward(data)






    @staticmethod
    def loadLayers(mlp_model_params):

        layers = [FullyConnectedLayer(pretrained=True, pretrained_weights=weights, pretrained_biases=biases, nonlinearity=nonlinearity, index=index) for (weights, biases, nonlinearity, index) in mlp_model_params.values()]

        # NOT including the input layer
        #output_layer_size = layers[-1].weights.size(dim=0)

        return layers


    @staticmethod
    def buildLayers(mlp_model_config, input_feature_count):

        neuron_counts = mlp_model_config.get("neuron_counts")
        activation_functions = mlp_model_config.get("activation_functions")
        # neuron_counts, activation_functions = mlp_model_config.values()

        neuron_counts.insert(0, input_feature_count)

        # NOT including the input layer
        num_layers = len(neuron_counts)
        #output_layer_size = neuron_counts[-1]

        layers = [FullyConnectedLayer(pretrained=False, input_count=neuron_counts[i], neuron_count=neuron_counts[i+1], nonlinearity=activation_functions[i], index=i+2) for i in range(num_layers-1)]

        return layers




    # def train(self, data, target, epochs=None):
    #
    #     epoch_plt = []
    #     loss_plt = []
    #
    #     if not epochs:
    #         epochs = data.size(dim=1)/self.batch_size
    #
    #     for epoch in range(1, int(epochs+1)):
    #
    #         loss = self.MSELoss(data, target)
    #         self.backprop(loss)
    #
    #         epoch_plt.append(epoch)
    #         loss_plt.append(loss.item())
    #         print(f"epoch = {epoch}, loss = {loss} ")
    #         print(f"_________________________________________")
    #
    #     self.saveParameters()
    #
    #     return epoch_plt, loss_plt



    def saveParameters(self):
        # for layer in self.layers:
        #     np.savetxt(f"layer_{layer.index}_weights_{layer.nonlinearity}.csv", layer.weights.detach().numpy(), delimiter=',', fmt='%f')
        #     np.savetxt(f"layer_{layer.index}_biases_{layer.nonlinearity}.csv", layer.biases.detach().numpy(), delimiter=',', fmt='%f')

        for layer in self.layers:
            torch.save(layer.weights, f"mlpParametersparameters/layer_{layer.index}_weights_{layer.nonlinearity}.pth")
            torch.save(layer.biases, f"mlpParametersparameters/layer_{layer.index}_biases_{layer.nonlinearity}.pth")



    # def reduce(self, x):
    #     if self.reduction == "mean":
    #         return x.mean()
    #     elif self.reduction == "sum":
    #         return x.sum()



    # def batch(self, data, target):
    #
    #     # sample batch
    #     batch_indicies = torch.randperm(n=data.size(dim=1))[:self.batch_size]  # stochastic
    #     #batch_indicies = torch.arange(start=0, end=self.batch_size)  # fixed
    #
    #     data_batch = data.T[batch_indicies].T
    #     target_batch = target.T[batch_indicies].T
    #
    #     return data_batch, target_batch




    # def BCELoss(self, data, target):
    #
    #     data_batch, target_batch = self.batch(data, target)
    #
    #     epsilon = 1e-4 #1e-10
    #     pred_batch = torch.clamp(self.forward(data_batch), epsilon, 1 - epsilon)
    #     errs = target_batch * torch.log(pred_batch) + (1-target_batch) * torch.log(1-pred_batch)
    #
    #     bce_loss = -(1/self.batch_size)*torch.sum(errs, dim=0)  # BCE (Binary Cross Entropy) Loss
    #     bce_loss_reduced = self.reduce(bce_loss)
    #
    #     return bce_loss_reduced




    # def MSELoss(self, data, target):
    #
    #     data_batch, target_batch = self.batch(data, target)
    #
    #     pred_batch = self.forward(data_batch)
    #     errs = (pred_batch - target_batch)**2
    #
    #     mse_loss = (1/self.batch_size)*torch.sum(errs, dim=0)  # MSE (Mean Square Error) Loss
    #     mse_loss_reduced = self.reduce(mse_loss)
    #
    #     return mse_loss_reduced





    def forward(self, curr_input):
        for layer in self.layers:
            curr_input = layer.feed(curr_input)
        return curr_input




    # def backprop(self, loss):
    #
    #     for layer in self.layers:
    #         layer.weights.grad = None
    #         layer.biases.grad = None
    #
    #     loss.backward()
    #
    #     with torch.no_grad():
    #         for layer in self.layers:
    #
    #             #if not torch.isnan(layer.weights).any():
    #             layer.weights -= self.learn_rate * layer.weights.grad
    #
    #             #if not torch.isnan(layer.biases).any():
    #             layer.biases -= self.learn_rate * layer.biases.grad
