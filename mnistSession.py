from CNN import CNN
from Data import fetchMLPParametersFromFile, fetchCNNParametersFromFile, plotTrainingResults, genOneHotEncodedMNISTStack, printMNISTInferenceResults






inference = False

if inference:

    datasetSize = 1000

    (imgBatch, labelBatch) = genOneHotEncodedMNISTStack(datasetSize, inference=inference)

    cnn = CNN(pretrained=True, cnn_model_params=fetchCNNParametersFromFile(), mlp_model_params=fetchMLPParametersFromFile())
    # for layer in cnn.layers:
    #     print(layer)

    predictionBatch = cnn.inference(imgBatch)

    printMNISTInferenceResults(dataset_size=datasetSize, img_batch=imgBatch, label_batch=labelBatch,
                               prediction_batch=predictionBatch)




else:

    isConvLayer =          [True, False, True, False]
    filterCounts =         [2, 2, 4, 4] # THE NUMBER OF FILTERS IN A POOLING KERNEL MUST MATCH THE NUMBER OF FILTERS IN THE PRECEEDING CONVOLUTIOAL LAYER KERNEL
    kernelShapes =         [(5, 5), (2, 2), (3, 3), (2, 2)]
    kernelStrides =        [1, 2, 1, 2]
    activationFunctions =  ["leakyReLU", "none", "sigmoid", "none"]

    CNNmodelConfig = {
        "is_conv_layer": isConvLayer,
        "filter_counts": filterCounts,
        "kernel_shapes": kernelShapes,
        "kernel_strides": kernelStrides,
        "activation_functions": activationFunctions
    }

    neuronCounts =        [10]
    activationFunctions = ["softmax"]

    MLPmodelConfig = {
        "neuron_counts": neuronCounts,
        "activation_functions": activationFunctions
    }




    datasetSize = 3

    (img_batch, label_batch) = genOneHotEncodedMNISTStack(datasetSize, inference=inference)

    # print(data_batch.size())
    # print(target_batch.size())


    cnn = CNN(pretrained=False, loss_func="CCELoss", cnn_model_config=CNNmodelConfig, input_data_dim=(1, 28, 28), mlp_model_config=MLPmodelConfig)

    (epochPlt, lossPlt) = cnn.train(img_batch, label_batch, epochs=datasetSize)

    plotTrainingResults(epochPlt, lossPlt)







