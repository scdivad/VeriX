import random
import torch
import numpy as np
import onnx
import onnxruntime as ort
from skimage.color import label2rgb
from matplotlib import pyplot as plt
from maraboupy import Marabou
import collections
import os
'''
passed tests
7270 425.6060001850128 165
passed tests
860 261.6677267551422 35
passed tests
5390 242.66076517105103 0
passed tests
5191 366.99401926994324 312
passed tests
5734 255.41195702552795 425
passed tests
6265 1750.8210852146149 186
passed tests
466 223.21297764778137 231
passed tests
4426 239.11786484718323 358
passed tests
5578 288.80060839653015 275
passed tests
8322 188.79589676856995 508
all:  2495 0
'''
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
                                    solveWithMILP=False)

    def __init__(self,
                 name,
                 dataset,
                 image,
                 model_path,
                 plot_original=True, 
                 # ONNX requires input of float32
                 threshold=1e-4):
        """
        To initialize the VeriX class.
        :param dataset: 'MNIST' or 'GTSRB'.
        :param image: an image array of shape (width, height, channel).
        :param model_path: the path to the neural network.
        :param plot_original: if True, then plot the original image.
        """
        self.dataset = dataset
        self.image = image
        self.name = name
        self.threshold = threshold
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
                    # TODO: also try different ablation methods.
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

            # difference = predictions - predictions_manip
            # features = difference[:, self.label]
            kl_difference = np.sum(predictions * np.log(np.maximum(predictions, 1e-9) / (np.maximum(predictions_manip, 1e-9))), axis=-1)
            features = kl_difference

            sorted_index = features.argsort() # TOOD: understand why this happens lmfao [::-1]

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

        '''
        u -> v iff u in A and v in {u}+B and v was in u's counterfactual
        note that it is impossible for there to be paths of length > 1
        '''
        c_G = [[] for _ in range(len(self.inputVars))]

        '''
        same_counterfactual has key of a counterfactual changed pixels in B  
        and values of pixels in A who utilized the same B values as their counterfactual
        '''
        same_counterfactual = {}

        unsat_set = set() # B
        sat_set = set() # A
        timeout_set = set()
        width, height, channel = self.image.shape[0], self.image.shape[1], self.image.shape[2]
        image = self.image.reshape(width * height, channel)

        tmp = self.inputVars.copy()

        dq = collections.deque(self.inputVars)

        counterfactuals = {}
        cnt_iter = 0
        while len(dq) > 0:
            pixel = dq.popleft()
            cnt_iter += 1
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
            # clear input and output constraints.
            self.mara_model.clearProperty()

            if exit_code == 'unsat':
                unsat_set.add(pixel)
            elif exit_code == 'TIMEOUT':
                timeout_set.add(pixel)
            elif exit_code == 'sat':
                sat_set.add(pixel)
                if True or plot_counterfactual:
                    counterfactuals[pixel] = [vals.get(i) for i in self.mara_model.inputVars[0].flatten()]
                    counterfactuals[pixel] = np.asarray(counterfactuals[pixel]).reshape(self.image.shape).astype(np.float32)

                    changed_mask = np.abs(counterfactuals[pixel].reshape(image.shape) - self.image.reshape(image.shape)) > 1e-6
                    c_G[pixel] = np.where(changed_mask)[0]
                    counterfactual_key = tuple(c_G[pixel])
                    same_counterfactual[counterfactual_key] = same_counterfactual.get(counterfactual_key, []) + [pixel]
                    if (len(same_counterfactual[counterfactual_key]) > 10 and changed_mask.sum() >= 5) or (len(same_counterfactual[counterfactual_key]) > 50 and changed_mask.sum() >= 1):
                        prediction = [vals.get(i) for i in self.outputVars]
                        prediction = np.asarray(prediction).argmax()
                        self.save_figure(image=counterfactual,
                                    path="axed_counterfactual-at-pixel-%d-predicted-as-%d.png" % (pixel, prediction),
                                    cmap="gray" if self.dataset == 'MNIST' else None)

                        # Remove elements in A that relied on counterfactual
                        for p in same_counterfactual[counterfactual_key]:
                            sat_set.remove(p)
                            counterfactuals.pop(p)
                            dq.append(p)
                        same_counterfactual[counterfactual_key] = []

                        # TODO: continue if p is the pixel
                        for p in c_G[pixel]:
                            # Remove elements in B of the counterfactual
                            unsat_set.remove(p)

                            # Remove elements from A that relied on that particular element in B
                            depends_on_p = {p2 for p2 in sat_set if p in c_G[p2]}
                            for p2 in depends_on_p:
                                dq.appendleft(p2)
                            sat_set = sat_set - depends_on_p
                        continue

                    prediction = [vals.get(i) for i in self.outputVars]
                    prediction = np.asarray(prediction).argmax()
                    # self.save_figure(image=counterfactual,
                    #             path="counterfactual-at-pixel-%d-predicted-as-%d.png" % (pixel, prediction),
                    #             cmap="gray" if self.dataset == 'MNIST' else None)
                    
        self.inputVars = tmp
        sat_set = list(sat_set)
        timeout_set = list(timeout_set)

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
        assert self.fast_test_explanation(image, epsilon, sat_set, unsat_set, counterfactuals)
        print("passed tests")
        return len(sat_set), len(timeout_set)
    

    def fast_test_explanation(self, image, epsilon, sat_set, unsat_set, counterfactuals):
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
            elif i in sat_set:
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
            else:
                return False
        for j in range(len(self.outputVars)):
            if j != self.label:
                self.mara_model.addInequality([self.outputVars[self.label], self.outputVars[j]],
                                                [1, -1], -self.threshold,
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
        
        image_flat = image.flatten()
        for i in sat_set:
            counterfactual_flat = counterfactuals[i].flatten()
            if np.linalg.norm(image_flat - counterfactual_flat, np.inf) > epsilon + self.threshold:
                print("330: ", np.linalg.norm(image_flat - counterfactual_flat, np.inf), epsilon + self.threshold)
                return False

            # make sure that different pixels from the original are entirely in the unsat set
            # except for one in the sat_set
            cnt_in_sat_set = 0
            for j in self.inputVars:
                if image_flat[j] != counterfactual_flat[j]:
                    if j in sat_set:
                        cnt_in_sat_set += 1
                    elif j in unsat_set:
                        pass
                    else:
                        print("343")
                        return False
            if cnt_in_sat_set > 1:
                print("346")
                return False
            # check if the counterfactual is valid by running the model
            counterfactual_result = self.onnx_session.run(None, {self.onnx_model.graph.input[0].name: np.expand_dims(counterfactuals[i], axis=0)})
            counterfactual_result = np.asarray(counterfactual_result[0])
            if counterfactual_result.argmax() == self.label:
                # print("sat pixel: ", i)
                np.set_printoptions(precision=10)
                # print("counterfactual_result: ", counterfactual_result)
                # print("self.label: ", self.label)
                # print("352")
                with open("warning.txt", "w+") as f:
                    f.write(f'{counterfactual_result[0].tolist()} {self.label}')
                    f.flush()
                
        return True
    
    def save_figure(self, image, path, cmap=None):
        """
        To plot figures.
        :param image: the image array of shape (width, height, channel)
        :param path: figure name.
        :param cmap: 'gray' if to plot gray scale image.
        :return: an image saved to the designated path.
        """
        folder = f"my_midpoint/{self.name}/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = folder + path
        
        with open(path + ".txt", 'w') as f:
            f.write(f'{image.flatten().tolist()}\n')

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


'''
for k in cnt_same_dep:
    # if key_cnt and k are 80% similar, remove k
    set_key = set(key_cnt)
    set_k = set(k)
    set_diff = set_k.difference(set_key)
    if len(set_diff) <= 0.2 * len(set_key):
        for p in cnt_same_dep[k]:
            sat_set.remove(p)
        self.inputVars = np.append(self.inputVars, np.array(cnt_same_dep[k]))
        cnt_same_dep[k] = []
        for p in k:
            unsat_set.remove(p)
        self.inputVars = np.append(self.inputVars, np.array(k))

continue

'''