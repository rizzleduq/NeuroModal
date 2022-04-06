from typing import Tuple, Union
import numpy as np
import cv2
from sklearn.datasets import load_digits
from nn_lib.data import Dataset
import matplotlib.pyplot as plt

DEFAULT_DATASET_PARAMETERS = dict(
    blobs=dict(centers=np.array([[0, 0], [3, 5]]), cluster_std=np.array([1, 2])),
    moons=dict(noise=0.1),
    circles=dict(noise=0.1, factor=0.55)
)


class DigitsMNISTDataset(Dataset):

    def __init__(self, train=True, **kwargs):
        """
        Create a dataset
        :param train: train or test images would be loaded
        """
        self.data_set = load_digits()

    def _stretch(self, image):
        return np.asarray(image).flatten() / 15.

    def _onehot_encode(self, label):
        result = np.zeros(10)
        result[label] = 1
        return result

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        data = self.data_set.data[index]
        target = self.data_set.target[index]
        result = self._stretch(data), self._onehot_encode(target)
        return result

    def __len__(self) -> int:
        return len(self.data_set.data)

    def _resize_image(self, image, width, height):
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    def vizualize_for_classifier(self, predictions: Union[np.ndarray, None] = None) -> None:

        width = 300
        height = 300
        for i in range(len(predictions)):
            tensor = self.data_set.images[i]
            id = self.data_set.target[i]
            image = self._resize_image(np.asarray(tensor), width, height)

            print("id", id, "", "predicted", predictions[i])

            plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
            plt.show()


if __name__ == '__main__':
    dataset = DigitsMNISTDataset()
    item = dataset[0]
    print(item)
