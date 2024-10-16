import numpy as np

class MinMaxScaler:

    """
    A class for scaling features to a specific range using min-max normalization.

    The MinMaxScaler scales each feature to a given range, typically [0, 1]. It performs
    the following transformations:
    - `fit`: Computes the minimum and maximum values for each feature.
    - `transform`: Scales the features based on the computed min and max values.
    - `inverse_transform`: Reverts the scaling to the original feature values.

    Attributes:
        diffs (list of float): List of differences between the max and min values for each feature.
        limits (list of tuple): List of tuples containing the min and max values for each feature.

    Methods:
        fit(data: np.ndarray):
            Computes the min and max values for each feature from the input data.

        transform(data: np.ndarray) -> np.ndarray:
            Scales the input data based on the min and max values computed during `fit`.

        inverse_transform(data: np.ndarray) -> np.ndarray:
            Reverts the scaled data back to its original feature values.

        lims():
            Prints the computed differences and limits for debugging purposes.
    """

    def __init__(self) -> None:
        self.diffs = []
        self.limits = []

    def lims(self):
        print(f"Diffs={self.diffs}")
        print(f"Limits={self.limits}")

    def fit(self, data: np.ndarray):
        transposed_data = np.transpose(np.copy(data))

        self.limits = [(np.min(_), np.max(_)) for _ in transposed_data]
        self.diffs = [(np.max(_) - np.min(_)) for _ in transposed_data]

    def transform(self, data: np.ndarray):
        transposed_data = np.transpose(np.copy(data))

        for idx in range(len(transposed_data)):
            transposed_data[idx] = ( transposed_data[idx] - self.limits[idx][0] ) / self.diffs[idx]

        return np.transpose(transposed_data)
    
    def inverse_transform(self, data: np.ndarray):
        transposed_data = np.transpose(np.copy(data))

        for idx in range(len(transposed_data)):
            transposed_data[idx] = transposed_data[idx] * self.diffs[idx] + self.limits[idx][0]

        return np.transpose(transposed_data)
    
class StandardScaler:

    """
    A class for standardizing features by removing the mean and scaling to unit variance.

    The StandardScaler can use either arithmetic or geometric means to compute the standard deviation
    and perform scaling. It performs the following transformations:
    - `fit`: Computes the means (arithmetic or geometric) and standard deviations for each feature.
    - `transform`: Standardizes the input data based on the computed means and standard deviations.
    - `inverse_transform`: Reverts the standardized data back to its original feature values.

    Attributes:
        arithmetic_means (list of float): List of arithmetic means for each feature.
        geometric_means (list of float): List of geometric means for each feature.
        standard_deviations (list of float): List of standard deviations for each feature.
        mean (str): Specifies which mean ("arithmetic" or "geometric") is used for scaling.

    Methods:
        fit(data: np.ndarray, mean="arithmetic"):
            Computes the means and standard deviations for each feature from the input data.
            The `mean` parameter determines whether to use arithmetic or geometric means.

        transform(data: np.ndarray) -> np.ndarray:
            Standardizes the input data based on the means and standard deviations computed during `fit`.

        inverse_transform(data: np.ndarray) -> np.ndarray:
            Reverts the standardized data back to its original feature values.

        lims():
            Prints the computed arithmetic means, geometric means, and standard deviations for debugging purposes.
    """

    def __init__(self) -> None:
        self.arithmetic_means = []
        self.geometric_means = []
        self.standard_deviations = []
        self.mean = ""

    def lims(self):
        print(f"Arithmetic Mean={self.arithmetic_means}")
        print(f"Geometric Means={self.geometric_means}")
        print(f"Standard Deviations={self.standard_deviations}")

    def fit(self, data: np.ndarray, mean="arithmetic"):
        transposed_data = np.transpose(np.copy(data))

        self.mean = mean

        self.arithmetic_means = [( np.sum(_) / len(_) ) for _ in transposed_data]
        self.geometric_means = [( np.prod(_) ** (1/len(_)) ) for _ in transposed_data]

        if self.mean == "arithmetic": 
            for index, row in enumerate(transposed_data):
                n = len(row)
                self.standard_deviations.append(
                    (np.sum([(item - self.arithmetic_means[index]) ** 2 for item in row]) / n ) ** .5
                )

        if self.mean == "geometric": 
            for index, row in enumerate(transposed_data):
                n = len(row)
                self.standard_deviations.append(
                    (np.sum([(item - self.geometric_means[index]) ** 2 for item in row]) / n ) ** .5
                )

    def transform(self, data: np.ndarray):
        transposed_data = np.transpose(np.copy(data))

        if self.mean == "arithmetic": 
            for idx in range(len(transposed_data)):
                transposed_data[idx] = ( transposed_data[idx] - self.arithmetic_means[idx] ) / self.standard_deviations[idx]

        if self.mean == "geometric":
            for idx in range(len(transposed_data)):
                transposed_data[idx] = ( transposed_data[idx] - self.geometric_means[idx] ) / self.standard_deviations[idx]

        return np.transpose(transposed_data)

    def inverse_transform(self, data: np.ndarray):
        transposed_data = np.transpose(np.copy(data))

        if self.mean == "arithmetic":
            for idx in range(len(transposed_data)):
                transposed_data[idx] = transposed_data[idx] * self.standard_deviations[idx] + self.arithmetic_means[idx]

        if self.mean == "geometric":
            for idx in range(len(transposed_data)):
                transposed_data[idx] = transposed_data[idx] * self.standard_deviations[idx] + self.geometric_means[idx]

        return np.transpose(transposed_data)