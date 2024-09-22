import matplotlib.pyplot as plt
import numpy as np

class Plotter(object):
    def __init__(self, title):
        self.x = np.array([])
        self.y = np.array([])
        self.title = title

    def init(self, title):
        self.__init__(title)

    def plot(self):
        plt.plot(self.x, self.y)
        plt.title(self.title)
        plt.xlabel("episode")
        plt.ylabel("score")
        plt.show()

    def append(self, x, y):
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)