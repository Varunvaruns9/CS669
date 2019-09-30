import pandas as pd
from calc import calc_mean_cov
from bayes import Bayes
import numpy as np


class DiagonalCommon(Bayes):
    def __init__(self, *args):
        super().__init__(*args)

        self.meanlist = []
        self.covlist = []
        covsum = np.zeros((2, 2))
        for train in self.trainlist:
            mean, cov = calc_mean_cov(train)
            self.meanlist.append(mean)
            covsum += cov
        covavg = np.diag(np.diag(covsum / len(self.trainlist)))
        covfinal = (np.sum(covavg) / covavg.shape[0]) * np.identity(covavg.shape[0])
        self.covlist = [covfinal] * len(self.trainlist)
