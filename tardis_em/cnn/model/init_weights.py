#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

from torch.nn import init

from tardis_em.utils.errors import TardisError


def weights_init_kaiming(m):
    """
    Initializes the weights of layers in a neural network module using the Kaiming
    initialization technique. This function checks the type of layer contained in
    the past module and applies different initialization strategies depending on
    the layer type. Specifically, it adjusts the weights and biases for convolutional
    layers, batch normalization layers, and group normalization layers. This function
    is commonly used for improving the convergence of deep learning models with
    ReLU activation functions.

    :param m: The module whose weights need to be initialized. It is expected
              to be an instance of a layer or a module class such as Conv2d,
              Conv3d, BatchNorm, or GroupNorm.

    :return: None
    """
    class_name = m.__class__.__name__

    if class_name.find("Conv3d") != -1:
        init.kaiming_normal_(tensor=m.weight.data)
    elif class_name.find("Conv2d") != -1:
        init.kaiming_normal_(tensor=m.weight.data)
    elif class_name.find("BatchNorm") != -1:
        init.normal_(tensor=m.weight.data, mean=1.0, std=0.02)
        init.constant_(tensor=m.bias.data, val=0.0)
    elif class_name.find("GroupNorm") != -1:
        init.normal_(tensor=m.weight.data, mean=1.0, std=0.02)
        init.constant_(tensor=m.bias.data, val=0.0)


def init_weights(net, init_type="kaiming"):
    """
    Initializes the weights of a neural network based on the specified initialization
    type. The function applies the kaiming initialization if `init_type` is set to
    "kaiming". If the provided `init_type` is not implemented, an error is raised.

    :param net: The neural network whose weights need to be initialized.
    :param init_type: A string specifying the type of weight initialization to be used.
        Defaults to "kaiming". Must be one of the implemented initialization methods.

    :return: None
    """
    if init_type == "kaiming":
        net.apply(weights_init_kaiming)
    else:
        TardisError(
            "140",
            "tardis_em/cnn/ini_weights.py",
            f"initialization method {init_type} is not implemented!",
        )
