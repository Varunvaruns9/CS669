import pandas as pd
from calc import calc_mean_cov
from bayes import Bayes
import numpy as np


class Common(Bayes):
    def __init__(self, *args):
        super().__init__(*args)

        self.meanlist = []
        self.covlist = []
        covsum = np.zeros((2, 2))
        for train in self.trainlist:
            mean, cov = calc_mean_cov(train)
            self.meanlist.append(mean)
            covsum += cov
        covsum = covsum / len(self.trainlist)
        self.covlist = [covsum] * len(self.trainlist)
