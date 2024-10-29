import random
import numpy as np
import onnx
import onnxruntime as ort
from skimage.color import label2rgb
from matplotlib import pyplot as plt
import os

# import sys
# sys.path.insert(0, "Marabou")
"""
After installing Marabou, load it from maraboupy.
"""
from maraboupy import Marabou
from maraboupy import MarabouUtils


class VeriX:
    """
    This is the VeriX class to take in an image and a neural network, and then output an explanation.
    """
    image = None
    keras_model = None
    mara_model = None
    traverse: str
    sensitivity = None
    dataset: str
    label: int
    inputVars = None
    outputVars = None
    epsilon: float
    """
    Marabou options: 'timeoutInSeconds' is the timeout parameter. 
    """
    options = Marabou.createOptions(numWorkers=16,
                                    timeoutInSeconds=300,
                                    verbosity=0,
                                    solveWithMILP=True)

    def __init__(self,
                 dataset,
                 name,
                 image,
                 model_path,
                 plot_original=True):
        """
        To initialize the VeriX class.
        :param dataset: 'MNIST' or 'GTSRB'.
        :param image: an image array of shape (width, height, channel).
        :param model_path: the path to the neural network.
        :param plot_original: if True, then plot the original image.
        """
        print(image)
        self.dataset = dataset
        self.name = name
        self.image = image
        """
        Load the onnx model.
        """
        self.onnx_model = onnx.load(model_path)
        self.onnx_session = ort.InferenceSession(model_path)
        prediction = self.onnx_session.run(None, {self.onnx_model.graph.input[0].name: np.expand_dims(image, axis=0)})
        prediction = np.asarray(prediction[0])
        self.label = prediction.argmax()
        """
        Load the onnx model into Marabou.
        Note: to ensure sound and complete analysis, load the model before the softmax activation function;
        if the model is trained from logits directly, then load the whole model. 
        """
        self.mara_model = Marabou.read_onnx(model_path)
        if self.onnx_model.graph.node[-1].op_type == "Softmax":
            mara_model_output = self.onnx_model.graph.node[-1].input
        else:
            mara_model_output = None
        self.mara_model = Marabou.read_onnx(filename=model_path,
                                            outputNames=mara_model_output)
        self.inputVars = np.arange(image.shape[0] * image.shape[1])
        self.outputVars = self.mara_model.outputVars[0].flatten()
        if plot_original:
            self.save_figure(image=image,
                        path=f"original-predicted-as-{self.label}.png",
                        cmap="gray" if self.dataset == 'MNIST' else None)

    def traversal_order(self,
                        traverse="heuristic",
                        plot_sensitivity=True,
                        seed=0):
        """
        To compute the traversal order of checking all the pixels in the image.
        :param traverse: 'heuristic' (by default) or 'random'.
        :param plot_sensitivity: if True, plot the sensitivity map.
        :param seed: if traverse by 'random', then set a random seed.
        :return: an updated inputVars that contains the traversal order.
        """
        self.traverse = traverse
        if self.traverse == "heuristic":
            width, height, channel = self.image.shape[0], self.image.shape[1], self.image.shape[2]
            temp = self.image.reshape(width * height, channel)
            image_batch = np.kron(np.ones(shape=(width * height, 1, 1), dtype=temp.dtype), temp)
            image_batch_manip = image_batch.copy()
            for i in range(width * height):
                """
                Different ways to compute sensitivity: use pixel reversal for MNIST and deletion for GTSRB.
                """
                if self.dataset == "MNIST":
                    image_batch_manip[i][i][:] = 1 - image_batch_manip[i][i][:]
                elif self.dataset == "GTSRB":
                    image_batch_manip[i][i][:] = 0
                else:
                    print("Dataset not supported: try 'MNIST' or 'GTSRB'.")
            image_batch = image_batch.reshape((width * height, width, height, channel))
            predictions = self.onnx_session.run(None, {self.onnx_model.graph.input[0].name: image_batch})
            predictions = np.asarray(predictions[0])
            image_batch_manip = image_batch_manip.reshape((width * height, width, height, channel))
            predictions_manip = self.onnx_session.run(None, {self.onnx_model.graph.input[0].name: image_batch_manip})
            predictions_manip = np.asarray(predictions_manip[0])
            difference = predictions - predictions_manip
            features = difference[:, self.label]
            sorted_index = features.argsort()
            self.inputVars = sorted_index
            self.sensitivity = features.reshape(width, height)
            if plot_sensitivity:
                self.save_figure(image=self.sensitivity, path=f'{self.dataset}-sensitivity-{self.traverse}.png')
        elif self.traverse == "random":
            random.seed(seed)
            random.shuffle(self.inputVars)
        else:
            print("Traversal not supported: try 'heuristic' or 'random'.")

    def get_explanation(self,
                        epsilon,
                        plot_explanation=True,
                        plot_counterfactual=False,
                        plot_timeout=False):
        """
        To compute the explanation for the model and the neural network.
        :param epsilon: the perturbation magnitude.
        :param plot_explanation: if True, plot the explanation.
        :param plot_counterfactual: if True, plot the counterfactual(s).
        :param plot_timeout: if True, plot the timeout pixel(s).
        :return: an explanation, and possible counterfactual(s).
        """
        unsat_set = []
        sat_set = []
        timeout_set = []
        width, height, channel = self.image.shape[0], self.image.shape[1], self.image.shape[2]
        image = self.image.reshape(width * height, channel)

        l1_vars = []
        for pixel in self.inputVars:
            for i in self.inputVars:
                """
                Set constraints on the input variables.
                """
                if i == pixel or i in unsat_set:
                    """
                    Set allowable perturbations on the current pixel and the irrelevant pixels.
                    """
                    if self.dataset == "MNIST":
                        self.mara_model.setLowerBound(i, max(0, image[i][:] - epsilon))
                        self.mara_model.setUpperBound(i, min(1, image[i][:] + epsilon))

                        # make fixed variable for the image pixel
                        e = MarabouUtils.Equation()
                        image_variable = self.mara_model.getNewVariable()
                        e.addAddend(1, image_variable)
                        e.setScalar(image[i][:])
                        self.mara_model.addEquation(e)

                        # find the difference between the perturbed pixel and the fixed variable
                        # diff_variable = perturbed_pixel - fixed_pixel
                        e2 = MarabouUtils.Equation()
                        diff_variable = self.mara_model.getNewVariable()
                        e2.addAddend(1, self.mara_model.inputVars[0][i])
                        e2.addAddend(-1, image_variable)
                        e2.addAddend(-1, diff_variable)
                        e2.setScalar(0)
                        self.mara_model.addEquation(e2)

                        # get the absolute value of the difference
                        abs_diff_variable = self.mara_model.getNewVariable()
                        self.mara_model.addAbsConstraint(diff_variable, abs_diff_variable)

                        l1_vars.append(abs_diff_variable)
                    elif self.dataset == "GTSRB":
                        self.mara_model.setLowerBound(3 * i, max(0, image[i][0] - epsilon))
                        self.mara_model.setUpperBound(3 * i, min(1, image[i][0] + epsilon))
                        self.mara_model.setLowerBound(3 * i + 1, max(0, image[i][1] - epsilon))
                        self.mara_model.setUpperBound(3 * i + 1, min(1, image[i][1] + epsilon))
                        self.mara_model.setLowerBound(3 * i + 2, max(0, image[i][2] - epsilon))
                        self.mara_model.setUpperBound(3 * i + 2, min(1, image[i][2] + epsilon))
                    else:
                        print("Dataset not supported: try 'MNIST' or 'GTSRB'.")
                else:
                    """
                    Make sure the other pixels are fixed.
                    """
                    if self.dataset == "MNIST":
                        self.mara_model.setLowerBound(i, image[i][:])
                        self.mara_model.setUpperBound(i, image[i][:])
                    elif self.dataset == "GTSRB":
                        self.mara_model.setLowerBound(3 * i, image[i][0])
                        self.mara_model.setUpperBound(3 * i, image[i][0])
                        self.mara_model.setLowerBound(3 * i + 1, image[i][1])
                        self.mara_model.setUpperBound(3 * i + 1, image[i][1])
                        self.mara_model.setLowerBound(3 * i + 2, image[i][2])
                        self.mara_model.setUpperBound(3 * i + 2, image[i][2])
                    else:
                        print("Dataset not supported: try 'MNIST' or 'GTSRB'.")

            # l1 norm constraint
            self.mara_model.addInequality(l1_vars, [1] * len(l1_vars), epsilon, isProperty=True)

            for j in range(len(self.outputVars)):
                """
                Set constraints on the output variables.
                """
                if j != self.label:
                    self.mara_model.addInequality([self.outputVars[self.label], self.outputVars[j]],
                                                  [1, -1], -1e-6,
                                                  isProperty=True)
                    exit_code, vals, stats = self.mara_model.solve(options=self.options, verbose=False)
                    """
                    additionalEquList.clear() is to clear the output constraints.
                    """
                    self.mara_model.additionalEquList.clear()
                    if exit_code == 'sat' or exit_code == 'TIMEOUT':
                        break
                    elif exit_code == 'unsat':
                        continue
            
            """
            clearProperty() is to clear both input and output constraints.
            """
            self.mara_model.clearProperty()
            """
            If unsat, put the pixel into the irrelevant set; 
            if timeout, into the timeout set; 
            if sat, into the explanation.
            """
            if exit_code == 'unsat':
                unsat_set.append(pixel)
            elif exit_code == 'TIMEOUT':
                timeout_set.append(pixel)
            elif exit_code == 'sat':
                sat_set.append(pixel)
                if plot_counterfactual:
                    counterfactual = [vals.get(i) for i in self.mara_model.inputVars[0].flatten()]
                    counterfactual = np.asarray(counterfactual).reshape(self.image.shape)
                    prediction = [vals.get(i) for i in self.outputVars]
                    prediction = np.asarray(prediction).argmax()
                    self.save_figure(image=counterfactual,
                                path="counterfactual-at-pixel-%d-predicted-as-%d.png" % (pixel, prediction),
                                cmap="gray" if self.dataset == 'MNIST' else None)
        if plot_explanation:
            mask = np.zeros(self.inputVars.shape).astype(bool)
            mask[sat_set] = True
            mask[timeout_set] = True
            plot_shape = self.image.shape[0:2] if self.dataset == "MNIST" else self.image.shape
            self.save_figure(image=label2rgb(mask.reshape(self.image.shape[0:2]),
                                        self.image.reshape(plot_shape),
                                        colors=[[0, 1, 0]] if self.traverse == 'heuristic' else [[1, 0, 0]],
                                        bg_label=0,
                                        saturation=1),
                        path="explanation-%d.png" % (len(sat_set) + len(timeout_set)))
        if plot_timeout:
            mask = np.zeros(self.inputVars.shape).astype(bool)
            mask[timeout_set] = True
            plot_shape = self.image.shape[0:2] if self.dataset == "MNIST" else self.image.shape
            self.save_figure(image=label2rgb(mask.reshape(self.image.shape[0:2]),
                                        self.image.reshape(plot_shape),
                                        colors=[[0, 1, 0]] if self.traverse == 'heuristic' else [[1, 0, 0]],
                                        bg_label=0,
                                        saturation=1),
                        path="timeout-%d.png" % len(timeout_set))
            
        assert self.fast_test_explanation(image, epsilon, sat_set, unsat_set)
        return len(sat_set), len(timeout_set)
    
    def fast_test_explanation(self, image, epsilon, sat_set, unsat_set, counterfactual):
        # check if there is a counterfactual with only the unsat pixels
        for i in self.inputVars:
            if i in unsat_set:
                if self.dataset == "MNIST":
                        self.mara_model.setLowerBound(i, max(0, image[i][:] - epsilon))
                        self.mara_model.setUpperBound(i, min(1, image[i][:] + epsilon))
                elif self.dataset == "GTSRB":
                    self.mara_model.setLowerBound(3 * i, max(0, image[i][0] - epsilon))
                    self.mara_model.setUpperBound(3 * i, min(1, image[i][0] + epsilon))
                    self.mara_model.setLowerBound(3 * i + 1, max(0, image[i][1] - epsilon))
                    self.mara_model.setUpperBound(3 * i + 1, min(1, image[i][1] + epsilon))
                    self.mara_model.setLowerBound(3 * i + 2, max(0, image[i][2] - epsilon))
                    self.mara_model.setUpperBound(3 * i + 2, min(1, image[i][2] + epsilon))
                else:
                    print("Dataset not supported: try 'MNIST' or 'GTSRB'.")
            else:
                if self.dataset == "MNIST":
                    self.mara_model.setLowerBound(i, image[i][:])
                    self.mara_model.setUpperBound(i, image[i][:])
                elif self.dataset == "GTSRB":
                    self.mara_model.setLowerBound(3 * i, image[i][0])
                    self.mara_model.setUpperBound(3 * i, image[i][0])
                    self.mara_model.setLowerBound(3 * i + 1, image[i][1])
                    self.mara_model.setUpperBound(3 * i + 1, image[i][1])
                    self.mara_model.setLowerBound(3 * i + 2, image[i][2])
                    self.mara_model.setUpperBound(3 * i + 2, image[i][2])
                else:
                    print("Dataset not supported: try 'MNIST' or 'GTSRB'.")
            for j in range(len(self.outputVars)):
                if j != self.label:
                    self.mara_model.addInequality([self.outputVars[self.label], self.outputVars[j]],
                                                  [1, -1], -1e-6,
                                                  isProperty=True)
                    exit_code, vals, stats = self.mara_model.solve(options=self.options, verbose=False)
                    self.mara_model.additionalEquList.clear() 
                    if exit_code == 'sat' or exit_code == 'TIMEOUT':
                        break
                    elif exit_code == 'unsat':
                        continue
            self.mara_model.clearProperty()
            assert exit_code != 'TIMEOUT'
            if exit_code == 'sat':
                return False
        # make sure that the norm difference from the original image is less than epsilon
        if np.linalg.norm(image.flatten() - counterfactual.flatten(), np.inf) > epsilon:
            return False
        # make sure that different pixels from the original are entirely in the unsat set
        # except for one in the sat_set
        cnt_in_sat_set = 0
        cnt_in_unsat_set = 0
        for i in self.inputVars:
            if image[i] != counterfactual[i]:
                if i in unsat_set:
                    cnt_in_unsat_set += 1
                elif i in sat_set:
                    cnt_in_sat_set += 1
                else:
                    return False
        if cnt_in_sat_set > 1:
            return False
        # check if the counterfactual is a valid counterfactual by running the model
        counterfactual_result = self.onnx_session.run(None, {self.onnx_model.graph.input[0].name: np.expand_dims(counterfactual, axis=0)})
        counterfactual_result = np.asarray(counterfactual_result[0])
        if counterfactual_result.argmax() == self.label:
            return False
        return True

            

    def test_explanation(self, image, epsilon, sat_set, unsat_set):
        for pixel in sat_set + [-1]: # pixel=-1 represents testing unsat pixels only
            for i in self.inputVars:
                if i == pixel or i in unsat_set:
                    if self.dataset == "MNIST":
                        self.mara_model.setLowerBound(i, max(0, image[i][:] - epsilon))
                        self.mara_model.setUpperBound(i, min(1, image[i][:] + epsilon))
                    elif self.dataset == "GTSRB":
                        self.mara_model.setLowerBound(3 * i, max(0, image[i][0] - epsilon))
                        self.mara_model.setUpperBound(3 * i, min(1, image[i][0] + epsilon))
                        self.mara_model.setLowerBound(3 * i + 1, max(0, image[i][1] - epsilon))
                        self.mara_model.setUpperBound(3 * i + 1, min(1, image[i][1] + epsilon))
                        self.mara_model.setLowerBound(3 * i + 2, max(0, image[i][2] - epsilon))
                        self.mara_model.setUpperBound(3 * i + 2, min(1, image[i][2] + epsilon))
                    else:
                        print("Dataset not supported: try 'MNIST' or 'GTSRB'.")
                else:
                    if self.dataset == "MNIST":
                        self.mara_model.setLowerBound(i, image[i][:])
                        self.mara_model.setUpperBound(i, image[i][:])
                    elif self.dataset == "GTSRB":
                        self.mara_model.setLowerBound(3 * i, image[i][0])
                        self.mara_model.setUpperBound(3 * i, image[i][0])
                        self.mara_model.setLowerBound(3 * i + 1, image[i][1])
                        self.mara_model.setUpperBound(3 * i + 1, image[i][1])
                        self.mara_model.setLowerBound(3 * i + 2, image[i][2])
                        self.mara_model.setUpperBound(3 * i + 2, image[i][2])
                    else:
                        print("Dataset not supported: try 'MNIST' or 'GTSRB'.")
            for j in range(len(self.outputVars)):
                if j != self.label:
                    self.mara_model.addInequality([self.outputVars[self.label], self.outputVars[j]],
                                                  [1, -1], -1e-6,
                                                  isProperty=True)
                    exit_code, vals, stats = self.mara_model.solve(options=self.options, verbose=False)
                    self.mara_model.additionalEquList.clear()
                    if exit_code == 'sat' or exit_code == 'TIMEOUT':
                        break
                    elif exit_code == 'unsat':
                        continue
            self.mara_model.clearProperty()
            assert exit_code != 'TIMEOUT'
            if not ((pixel < 0 and exit_code == 'unsat') or 
                    (pixel >= 0 and exit_code == 'sat')
                ):
                return False

        return True

    def save_figure(self, image, path, cmap=None):
        return
        """
        To plot figures.
        :param image: the image array of shape (width, height, channel)
        :param path: figure name.
        :param cmap: 'gray' if to plot gray scale image.
        :return: an image saved to the designated path.
        """
        folder = f"outputs/{self.name}/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = folder + path
        
        fig = plt.figure()
        ax = plt.Axes(fig, [-0.5, -0.5, 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        if cmap is None:
            plt.imshow(image)
        else:
            plt.imshow(image, cmap=cmap)
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)