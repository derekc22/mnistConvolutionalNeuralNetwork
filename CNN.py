import torch
import numpy as np
from ConvolutionalLayer import ConvolutionalLayer
from MLP import MLP
import torch.nn as nn


class CNN:

    def __init__(self, pretrained, **kwargs):

        self.learn_rate = 0.01
        self.batch_size = 1
        self.reduction = "mean"


        if not pretrained:
            self.loss_func = kwargs.get("loss_func")
            self.layers = CNN.buildLayers(cnn_model_config=kwargs.get("cnn_model_config"))

            MLP_input_feature_count = self.calcMLPInputSize(kwargs.get("input_data_dim"))
            self.MLP = MLP(pretrained=False, input_feature_count=MLP_input_feature_count, mlp_model_config=kwargs.get("mlp_model_config"))

        else:
            self.layers = CNN.loadLayers(kwargs.get("cnn_model_params"))
            self.MLP = MLP(pretrained=True, mlp_model_params=kwargs.get("mlp_model_params"))





    def calcMLPInputSize(self, input_data_dim):

        input_data_dim = (1, ) + input_data_dim

        dummy_data = torch.empty(size=input_data_dim)
        dummy_MLP_input = self.forward(dummy_data, dummy=True)
        dummy_MLP_input_feature_count = dummy_MLP_input.size(dim=0)

        return dummy_MLP_input_feature_count





    def inference(self, data):
        return self.forward(data)



    @staticmethod
    def loadLayers(cnn_model_params):

        layers = [ConvolutionalLayer(pretrained=True, is_conv_layer=(is_conv_layer=="True"), pretrained_kernels=kernels, pretrained_biases=biases, nonlinearity=nonlinearity, kernel_stride=stride, index=index) for (is_conv_layer, kernels, biases, nonlinearity, stride, index) in cnn_model_params.values()]

        return layers


    @staticmethod
    def buildLayers(cnn_model_config):

        is_conv_layer = cnn_model_config.get("is_conv_layer")
        filter_counts = cnn_model_config.get("filter_counts")
        kernel_shapes = cnn_model_config.get("kernel_shapes")
        kernel_strides = cnn_model_config.get("kernel_strides")
        activation_functions = cnn_model_config.get("activation_functions")

        num_layers = len(is_conv_layer)

        layers = [ConvolutionalLayer(pretrained=False, is_conv_layer=is_conv_layer[i], filter_count=filter_counts[i], kernel_shape=kernel_shapes[i], kernel_stride=kernel_strides[i], nonlinearity=activation_functions[i], index=i+1) for i in range(num_layers)]

        return layers






    def train(self, data, target, epochs=None):

        epoch_plt = []
        loss_plt = []

        # if not epochs:
        #     epochs = data.size(dim=1)/self.batch_size

        for epoch in range(1, int(epochs+1)):

            data_batch, target_batch = self.batch(data, target)
            pred_batch = self.forward(data_batch)

            #loss = self.(pred_batch, target_batch)
            loss = getattr(self, self.loss_func)(pred_batch, target_batch)
            self.backprop(loss)

            epoch_plt.append(epoch)
            loss_plt.append(loss.item())
            print(f"epoch = {epoch}, loss = {loss} ")
            print(f"_________________________________________")


        self.saveParameters()
        self.MLP.saveParameters()

        return epoch_plt, loss_plt





    def saveParameters(self):

        for layer in self.layers:

            torch.save(layer.kernels, f"cnnParameters/cnn_layer_{layer.index}_kernels_{layer.nonlinearity}_{layer.is_conv_layer}_{layer.kernel_stride}.pth")
            torch.save(layer.biases, f"cnnParameters/cnn_layer_{layer.index}_biases_{layer.nonlinearity}_{layer.is_conv_layer}_{layer.kernel_stride}.pth")




    def reduce(self, x):
        if self.reduction == "mean":
            return x.mean()
        elif self.reduction == "sum":
            return x.sum()




    def batch(self, data, target):

        # sample batch
        batch_indicies = torch.randperm(n=data.size(dim=0))[:self.batch_size]  # stochastic
        #batch_indicies = torch.arange(start=0, end=self.batch_size)  # fixed


        data_batch = data[batch_indicies]
        # print("data_batch = ")
        # print(data_batch)


        # print("batch indicies = ")
        # print(batch_indicies)

        target_batch = target.T[batch_indicies].T
        # target_batch = target[batch_indicies]

        # print("target_batch = ")
        # print(target_batch)
        # print("target = ")
        # print(target)


        return data_batch, target_batch



    def CCELoss(self, pred_batch, target_batch):

        epsilon = 1e-8
        pred_batch = torch.clamp(pred_batch, epsilon, 1 - epsilon)
        errs = torch.mul(target_batch, torch.log(pred_batch))

        cce_loss = -torch.sum(errs, dim=0)  # CCE (Categorical Cross Entropy) Loss
        cce_loss_reduced = self.reduce(cce_loss)

        return cce_loss_reduced



    def BCELoss(self, pred_batch, target_batch):

        epsilon = 1e-3
        pred_batch = torch.clamp(pred_batch, epsilon, 1 - epsilon)
        errs = target_batch * torch.log(pred_batch) + (1-target_batch) * torch.log(1-pred_batch)

        bce_loss = -(1/self.batch_size)*torch.sum(errs, dim=0)  # BCE (Binary Cross Entropy) Loss
        bce_loss_reduced = self.reduce(bce_loss)

        # print("pred = ")
        # print(pred_batch)
        # print(pred_batch.size())
        # print("target = ")
        # print(target_batch)
        # print(target_batch.size())
        # print("errs = ")
        # print(errs)
        # print(errs.size())

        return bce_loss_reduced



    def MSELoss(self, pred_batch, target_batch):

        errs = (pred_batch - target_batch)**2
        mse_loss = (1/self.batch_size)*torch.sum(errs, dim=0)  # MSE (Mean Square Error) Loss
        mse_loss_reduced = self.reduce(mse_loss)

        return mse_loss_reduced




    # def CCELoss(self, data, target):
    #     criterion = nn.CrossEntropyLoss()
    #
    #     pred = self.forward(data).reshape(1, -1)
    #     target = torch.tensor([torch.nonzero(target == 1.0)[:, 0].item()])
    #     return criterion(pred, target)




    def forward(self, curr_input, dummy=False):  # maybe recursion would work for this?

        for layer in self.layers:
            if layer.is_conv_layer:
                curr_input = layer.convolve(curr_input)
            else:
                curr_input = layer.maxpool(curr_input)

        curr_input_batch_size = curr_input.size(dim=0)
        flattened_feature_map = curr_input.view(curr_input_batch_size, -1).to(torch.float64).T
        """flattened_feature_map = curr_input.reshape(-1, curr_input_batch_size).to(torch.float64)""" # <------------THIS LINE OF CODE WAS THE PROBLEM

        # print(f"flattened_feature_map_size = {flattened_feature_map.size()}")

        if dummy:
            return flattened_feature_map
        else:
            return self.MLP.forward(flattened_feature_map)






    def backprop(self, loss):

        for layer in self.layers:
            if layer.is_conv_layer:
                layer.kernels.grad = None
                layer.biases.grad = None

        for layer in self.MLP.layers:
            layer.weights.grad = None
            layer.biases.grad = None

        loss.backward()

        with torch.no_grad():
            for layer in self.layers:
                if layer.is_conv_layer:

                    if layer.index == 0:
                        # pass
                        # print(layer.kernels)
                        print(layer.kernels.grad)


                    #if not torch.isnan(layer.weights).any():
                    layer.kernels -= self.learn_rate * layer.kernels.grad

                    #if not torch.isnan(layer.biases).any():
                    layer.biases -= self.learn_rate * layer.biases.grad


            for layer in self.MLP.layers:
                #if not torch.isnan(layer.weights).any():
                layer.weights -= self.MLP.learn_rate * layer.weights.grad

                #if not torch.isnan(layer.biases).any():
                layer.biases -= self.MLP.learn_rate * layer.biases.grad



