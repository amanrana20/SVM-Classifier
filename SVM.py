import math
import random


class TrainingData:
    # Private class variables
    __filename = 'breast-cancer-wisconsin.data.txt'
    __file_data = []

    # public class variable
    data = []

    def __init__(self):
        file_data = open(self.__filename, 'r')

        for line in file_data:
            line = (line.strip('\n'))
            line = line.split(',')
            line_data = []

            for number in line:
                if number != '?':
                    number = int(number)
                else:
                    number = 0

                line_data.append(number)
            self.__file_data.append(line_data)

    def get_data(self):
        """This function gets the training data into X and y format.
        The data is stored in the format -> X, y, alpha"""

        for item in self.__file_data:
            # initializing X and y to [] and y at the starting of each loop
            X = []

            for i in range(1, len(item) - 1):
                X.append(item[i])

            if item[-1] == 2:
                y = 1
            else:
                y = -1

            self.data.append([X, y, 0])

        # print('Reshuffling the database...')
        # random.shuffle(self.data)

        return self.data


class Kernel:

    def gaussian_kernel(self, X1, X2, sigma=8):  # Sigma=8 best till now
        """ Gaussian kernel """
        sum = 0

        for i in range(len(X1)):
            sum += math.exp(- ( ((X1[i] - X2[i]) ** 2) / (2 * (sigma ** 2)) ) )

        return sum


class SVM:

    # Class Variables
    data = []  # The data is stored in the format -> X, y, alpha
    __training_data = TrainingData()
    b = 0

    def __init__(self, training_data_length, length_of_data_to_check_for_accuracy):
        self.data = self.__training_data.get_data()  # Send true to print the data else False
        self.training_data_length = training_data_length
        self.length_of_data_to_check_for_accuracy = length_of_data_to_check_for_accuracy
        self.optimize()
        print('Checking accuracy of SVM...')
        self.checkAccuracy()

    def print_data(self):
        counter = 0

        for i in range(self.training_data_length):
            item = self.data[i]

            if -1e-5 < item[2] < 1:
                item[2] = 0
                print(item)
            else:
                print(item)

                counter += 1

        print('<<<< NON-ZERO TRAINING COUNTER: %s/%s >>>>' % (counter, self.training_data_length))

    def optimize(self):
        print('Optimizing Lagrangian Multipliers...')
        self.data, self.b = SMO(self.data).smo(self.training_data_length)
        print('Finished Optimizing... 100%')
        # self.print_data()
        # print('Printed Data...')

    def checkAccuracy(self):
        counter = 0

        for j in range(self.training_data_length, len(self.data)):
            x = self.data[j][0]
            y = self.data[j][1]
            fx = 0

            # Calculate sum(alpha_i * yi * K(xi.x) + b) and check for sign
            for i in range(desired_training_data_length):

                # Getting all the variables related to i
                xi = self.data[i][0]
                yi = self.data[i][1]
                alpha_i = self.data[i][2]

                # Calculating fx and summing it up for each iteration of j
                fx += (alpha_i * yi * Kernel().gaussian_kernel(x, xi)) + self.b

            # print(fx, y)
            if (fx < 0 and y < 0) or (fx > 0 and y > 0):
                counter += 1

        print('\n<< Accuracy: %s' % ((counter * 100) / self.length_of_data_to_check_for_accuracy), '% >>')


class SMO:
    data = []
    C = 7  # C is the Regularization parameter i.e. 0 < alpha < C
    tolerance = 0.1
    b = 0
    b_avg = 0

    def __init__(self, data):
        self.data = data
        self.maximum_passes = 150

    def smo(self, training_length):
        initial_pass = 0
        sum_b = 0
        counter = 0

        while initial_pass < self.maximum_passes:
            num_of_changed_alphas = 0

            for i in range(training_length):

                item_i = self.data[i]
                x_i = item_i[0]
                y_i = item_i[1]
                alpha_old_i = item_i[2]
                Ei = ((alpha_old_i * item_i[1] * Kernel().gaussian_kernel(x_i, x_i)) + self.b) - y_i

                if (item_i[1] * Ei < -self.tolerance and alpha_old_i < self.C) or (item_i[1] * Ei > self.tolerance and alpha_old_i > 0):

                    #  Ensuring j is not equal to i after the while statement is over
                    j = i
                    while j == i:
                        j = random.randint(0, training_length-1)

                    # Saving old alphas
                    item_j = self.data[j]
                    x_j = item_j[0]
                    y_j = item_j[1]
                    alpha_old_j = item_j[2]

                    # Calculating Ej
                    """Ej = alpha * y * kernel"""
                    Ej = ((alpha_old_j * item_j[1] * Kernel().gaussian_kernel(x_i, x_j)) + self.b) - y_j

                    # Computing L and H
                    if y_i == y_j:
                        L = max(0, alpha_old_i + alpha_old_j - self.C)
                        H = min(self.C, alpha_old_i + alpha_old_j)
                    else:
                        L = max(0, alpha_old_j - alpha_old_i)
                        H = min(self.C, self.C + alpha_old_j - alpha_old_i)

                    # if L >= H, DO NOT GO FURTHER and go to the next iteration
                    if H > L:

                        # Calculate n
                        n = Kernel().gaussian_kernel(x_i, x_i) + Kernel().gaussian_kernel(x_j, x_j) - 2 * Kernel().gaussian_kernel(x_i, x_j)

                        # If n > 0, proceed, otherwise go to the next iteration.
                        if n > 0:

                            # Calculating new alpha_J
                            alpha_new_j = alpha_old_j + ((y_j * (Ei - Ej))/n)

                            # Constraining the new alpha_j to lie in between L and H
                            if alpha_new_j > H:
                                alpha_new_j = H

                            if alpha_new_j < L:
                                alpha_new_j = L

                            # UPDATING the values of alpha_j in the data-set
                            item_j[2] = alpha_new_j

                            if abs(alpha_new_j - alpha_old_j) >= 1e-5:

                                # Calculating alpha_new_I
                                alpha_new_i = alpha_old_i + y_i * y_j * (alpha_old_j - alpha_new_j)

                                # UPDATING the values of alpha_i in the data-set
                                item_i[2] = alpha_new_i

                                # Computing b1 and b2
                                b1 = self.b - Ei - y_i*(alpha_new_i - alpha_old_i)*Kernel().gaussian_kernel(x_i, x_i) - item_j[1]*(alpha_new_j - alpha_old_j)*Kernel().gaussian_kernel(x_i, x_j)
                                b2 = self.b - Ej - y_i*(alpha_new_i - alpha_old_i)*Kernel().gaussian_kernel(x_i, x_j) - item_j[1]*(alpha_new_j - alpha_old_j)*Kernel().gaussian_kernel(x_j, x_j)

                                # Computing b
                                if 0 < alpha_new_i < self.C:
                                    self.b = b1

                                else:
                                    if 0 < alpha_new_j < self.C:
                                        self.b = b2

                                    else:
                                        self.b = (b1 + b2)/2
                                # Counting the number of iterations that reach till here
                                counter += 1

                                # Computing the average of all b
                                sum_b += self.b

                                # Increment the value of num_of_changed_alphas by 1
                                num_of_changed_alphas += 1

                # if condition for KKT

            # for loop

            initial_pass += 1

        # while loop

        # Computing b_avg
        self.b_avg = sum_b / counter

        # Return changed data
        return self.data, self.b_avg


"""<<< Can change this >>>"""
desired_training_data_length = 400  # The rest of the data is used for testing purposes
"""<<< Can change this >>>"""

"""DO NOT CHANGE"""
length_of_data_to_check_for_checking_accuracy = 700 - desired_training_data_length
svm = SVM(desired_training_data_length, length_of_data_to_check_for_checking_accuracy)
"""DO NOT CHANGE"""
