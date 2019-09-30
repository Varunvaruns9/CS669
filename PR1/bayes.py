import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Bayes:
    def __init__(self, *args):
        self.trainlist = []
        self.testlist = []
        self.data = []
        for arg in args:
            np.random.shuffle(arg)
            self.data.append(arg)
            split = int(len(arg)*0.75)
            train = arg[:split]
            test = arg[split:]
            self.trainlist.append(train)
            self.testlist.append(test)

    def predict(self, x):
        p = []
        for cov, mean in zip(self.covlist, self.meanlist):
            p.append((1 / np.sqrt(pow(2 * np.pi, 2) * np.linalg.det(cov))) * np.exp(-0.5 * np.matmul(np.matmul(x - mean, np.linalg.inv(cov)), x - mean)))
        for j, _ in enumerate(p):
            p[j] = p[j] * len(self.trainlist[j])
        maxval = max(p)
        maxind = p.index(maxval)
        return maxind

    def result(self):
        total = 0
        totalcorrect = 0
        meanprecision = 0
        meanrecall = 0
        meanfmeasure = 0
        confusion = np.zeros((len(self.testlist), len(self.testlist)))
        for i, test in enumerate(self.testlist):
            correct = 0
            for row in test:
                predicted = self.predict(row)
                confusion[i][predicted] += 1
                if predicted == i:
                    correct += 1
            total += len(test)
            totalcorrect += correct
            # print("Class{}: {}/{}".format(i+1, totalcorrect, total))
        for i, _ in enumerate(self.testlist):
            totalpos = 0
            totalpred = 0
            for j, _ in enumerate(self.testlist):
                totalpos += confusion[j][i]
                totalpred += confusion[i][j]
            precision = confusion[i][i] / totalpos
            recall = confusion[i][i] / totalpred
            fmeasure = 2*precision*recall/(precision+recall)
            print("Precision for Class {}: {}".format(i, precision))
            print("Recall for Class {}: {}".format(i, recall))
            print("F-measure for Class {}: {}".format(i, fmeasure))
            meanprecision += precision / len(self.testlist)
            meanrecall += recall / len(self.testlist)
            meanfmeasure += fmeasure / len(self.testlist)

        print("Confusion Matrix:")
        print("(Row - True, Column - Predicted)")
        print(pd.DataFrame(confusion, dtype=int))
        print("Mean Precision: {}".format(meanprecision))
        print("Mean Recall: {}".format(meanrecall))
        print("Mean F-measure: {}".format(meanfmeasure))
        print("Accuracy: {}".format(totalcorrect / total))

    def plot(self):
        self.fig = plt.figure()
        xmax = -1e9
        xmin = 1e9
        ymax = -1e9
        ymin = 1e9
        for val in self.trainlist:
            xmax = max(xmax, val[:,0].max())
            xmin = min(xmin, val[:,0].min())
            ymax = max(ymax, val[:,1].max())
            ymin = min(ymin, val[:,1].min())
        for i, val in enumerate(self.trainlist):
            plt.scatter(val[:,0], val[:,1], label="Class {}".format(i))
        hx = (xmax - xmin) / 100
        hy = (ymax - ymin) / 100
        x_min, x_max = xmin - 20*hx, xmax + 20*hx
        y_min, y_max = ymin - 20*hy, ymax + 20*hy
        xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))
        Z = np.array([self.predict(x) for x in np.c_[xx.ravel(), yy.ravel()]])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.25)
        plt.contour(xx, yy, Z, colors='k', linewidths=0.5)
        plt.legend()
        plt.show()
