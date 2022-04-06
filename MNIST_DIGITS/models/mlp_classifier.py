from typing import Union, Tuple, List
import numpy as np

from nn_lib import Tensor
from nn_lib.mdl import Module, Linear


class MLPClassifier(Module):
    """
    Class representing a multilayer perceptron network for solving classification task
    """
    def __init__(self, in_features: int, number_of_classes: int,
                hidden_layer_sizes: Union[Tuple[int, ...], List[int]],
                build_layers: bool = True):
        """
        Creates binary MLP classifier
        :param in_features: number of feature in the input data
        :param hidden_layer_sizes: number of neurons in hidden layers of MLP
        """
        self.in_features = in_features
        self.number_of_classes = number_of_classes
        self.hidden_layer_sizes = list(hidden_layer_sizes)

        self._parameters = []
        self.layers = []        # type: List[Linear]

        if build_layers:
            self._build_layers()

    def parameters(self) -> List[Tensor]:
        result = [parameter for parameter in self._parameters if parameter.requires_grad]
        return result

    def _get_activation(self, number_of_layers: int, layer_number: int) -> str:
        return 'relu' if number_of_layers == layer_number else 'none'

    def _build_layers(self) -> None:
        """
        Create hidden layers of the MLP
        First linear layer transforms input data into self.hidden_layer_sizes[0] dimensions
        Last linear layer transforms features from self.hidden_layer_sizes[-1] dimensions into a single dimension with
        no activation function
        Output of the last layer will later be used as an argument to the sigmoid function
        :return: None
        """
        number_of_layers = len(self.hidden_layer_sizes)
        if number_of_layers == 0:
            return

        for i in range(0, number_of_layers):
            in_dim = out_dim if i != 0 else self.in_features
            out_dim = self.hidden_layer_sizes[i]

            self._add_layer(in_dim, out_dim, 'relu')

        self._add_layer(out_dim, self.number_of_classes, 'none')


    def _add_layer(self, in_dim: int, out_dim: int, activation_fn: str) -> None:
        """
        Adds a single layer to the network and updates internal list of parameters accordingly (both weight and bias)
        :param in_dim: number of features returned by the previous layer
        :param out_dim: number of features for the added layer to return
        :param activation_fn: activation function to apply to the outputs of the added layer
        :return: None
        """
        layer = Linear(in_dim, out_dim, activation_fn)
        self.layers.append(layer)
        self._parameters.append(layer.weight)
        self._parameters.append(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Pass an input through the network layers obtaining the prediction logits; later still need to apply
        sigmoid function to obtain confidence values from [0, 1]
        :param x: input data batch of the shape (B, self.in_features)
        :return: prediction batch of logits of the shape (B,)
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def parameter_count(self) -> int:
        """
        Count total number of trainable parameters of the network
        :return: number of trainable parameters of the network
        """
        result = 0
        for param in self.parameters():
            result += np.prod(param.shape)
        return result

    def __str__(self) -> str:
        result = '\n'.join(map(str, self.layers)) + f'\nTotal number of parameters: {self.parameter_count()}'
        return result





