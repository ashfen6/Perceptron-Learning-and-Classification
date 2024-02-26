"""binary_perceptron.py
One of the starter files for use in CSE 415, Autumn 2023
Assignment 6.
Complete this python file.

This program can be run from the given Python program
called run_2_class_2_feature_iris_data.py.
"""


def student_name():
    return "Ashley Fenton and Alex Pullen"  # Replaced with my own name


class BinaryPerceptron:
    """
    Class representing the Binary Perceptron
    ---
    It is an algorithm that can learn a binary classifier
    """
    
    def __init__(self, weights=None, alpha=0.5):
        """
        Initialize the Binary Perceptron
        ---
        weights: Weight vector of the form [w_0, w_1, ..., w_{n-1}, bias_weight]
        alpha: Learning rate
        """
        if weights is None:
            self.weights = [0, 0, 0]
        else:
            self.weights = weights[:]
        self.alpha = alpha
    
    def classify(self, x_vector):
        """
        Method that classifies a given data point into one of 2 classes.
        ---
        Inputs:
        x_vector = [x_0, x_1, ..., x_{n-1}]
        Note: y (correct class) is not part of the x_vector.

        Returns:
        y_hat: Predicted class
              +1 if the current weights classify x_vector as positive i.e. the required dot product must be >=0,
        else  -1 if it is classified as negative.
        """
        #Computing the dot product of the weights and the input vector called 'x_vector'
        dot_product = sum(w*x for w, x in zip(self.weights[:-1], x_vector)) + self.weights[-1]
        #If the dot product is greater than or equal to 0, it will return a +1 (indicating a positive classification)
        #Otherwise, it will return a -1 (indicating a negative classification)
        y_hat = +1 if dot_product >= 0 else -1
        return y_hat
    
    def train_with_one_example(self, x_vector, y):
        """
        Method that updates the model weights using a particular training example (x_vector,y)
        and returns whether the model weights were actually changed or not
        ---
        Inputs:
        x_vector: Feature vector, same as method classify
        y: Actual class of x_vector
            +1 if x_vector represents a positive example,
        and -1 if it represents a negative example.
        Returns:
        weight_changed: True if there was a change in the weights
                        else False
        """
        #Predicting the class of x_vector using the current weights
        y_hat = self.classify(x_vector)
        #Initialzing a flag to keep track of whether the weights are changed throughout the method
        weight_changed = False
        #Updating the weights according to the perception learning rules 
        #if the predicted class (y_hat) does not match the actual class (y)
        if y != y_hat:
            #Iterating over each fature in x_vector (excluding the bias term) to update its weight
            for i in range(len(self.weights) - 1):
                #Updating the weight for each feature according to the perceptron learning rule:
                #weight_new=weight_old+learning_rate*(desired_output-predicted_output)*feature_value
                #self.alpha is the learning rate
                #(y * x_vector[i]) is the product of the actual class and the feature value
                self.weights[i] += self.alpha * (y * x_vector[i])
            #Updating the bias weight
            self.weights[-1] += self.alpha * y 
            #Now the weights have been updated, change the flag to true 
            weight_changed = True
        return weight_changed
    
    def train_for_an_epoch(self, training_data):
        """
        Method that goes through the given training examples once, in the order supplied,
        passing each one to train_with_one_example.
        ---
        Input:
        training_data: Input training data
        [[x_vector_1, y_1], [x_vector_2, y_2], ...]
        where each x_vector is concatenated with the corresponding y value.

        Returns:
        changed_count: Return the number of weight updates.
        (If zero, then training has converged.)
        """
        #Keeping track of the number of times the weights are updated
        changed_count = 0
        #Looping through each example in the training data
        for example in training_data:
            #Extracting the feature vector from the current example
            x_vector = example[:-1]  
            #Extracting the label (y) for the current example
            y = example[-1]
            #Incrementing the changed_count by 1 if the weights were updated
            if self.train_with_one_example(x_vector, y):
                changed_count += 1
        #Returning the number of times the model adjusted its weights during this epoch
        return changed_count


def sample_test():
    """
    May be useful while developing code
    Trains the binary perceptron using a synthetic training set
    Prints the weights obtained after training
    """
    DATA = [
        [-2, 7, +1],
        [1, 10, +1],
        [3, 2, -1],
        [5, -2, -1]]
    bp = BinaryPerceptron()
    print("Training Binary Perceptron for 3 epochs.")
    for i in range(3):
        bp.train_for_an_epoch(DATA)
    print("Binary Perceptron weights:")
    print(bp.weights)
    print("Done.")


if __name__ == '__main__':
    sample_test()
