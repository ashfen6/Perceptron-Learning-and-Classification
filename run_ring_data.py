from binary_perceptron import BinaryPerceptron  # Your implementation of binary perceptron
from plot_bp import PlotBinaryPerceptron
import csv  # For loading data.
from matplotlib import pyplot as plt  # For creating plots.
import remapper


class PlotRingBP(PlotBinaryPerceptron):
    """
    Plots the Binary Perceptron after training it on the Iris dataset
    ---
    Extends the class PlotBinaryPerceptron
    """

    def __init__(self, bp, plot_all=False, n_epochs=25, IS_REMAPPED=True):
        super().__init__(bp, plot_all, n_epochs)  # Calls the constructor of the super class
        self.IS_REMAPPED = IS_REMAPPED

    def read_data(self):
        """
        Read data from the Ring dataset with 2 features and 2 classes
        for both training and testing.
        ---
        Overrides the method in PlotBinaryPerceptron
        """
        data_as_strings = list(csv.reader(open('ring-data.csv'), delimiter=','))
        if self.IS_REMAPPED:
            self.TRAINING_DATA = [
                [
                    remapper.remap(float(f1), float(f2))[0],
                    remapper.remap(float(f1), float(f2))[1],
                    int(c)
                ]
                for [f1, f2, c] in data_as_strings
            ]
        else:
            self.TRAINING_DATA = [[float(f1), float(f2), int(c)] for [f1, f2, c] in data_as_strings]

        self.TESTING_DATA = self.TRAINING_DATA.copy()

    def plot(self):
        """
        Plots the dataset as well as the binary classifier
        ---
        Overrides the method in PlotBinaryPreceptron
        """
        plt.title("Iris setosa (blue) vs iris versicolor (red)")
        plt.xlabel("Sepal length")
        plt.ylabel("Petal length")
        plt.legend(loc='best')
        plt.show()


if __name__ == '__main__':
    binary_perceptron = BinaryPerceptron(alpha=0.5)
    pbp = PlotRingBP(binary_perceptron)
    pbp.train()
    pbp.test()
    pbp.plot()
