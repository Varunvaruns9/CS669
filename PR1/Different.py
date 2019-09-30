import pandas as pd
from calc import calc_mean_cov
from bayes import Bayes
import numpy as np


class Different(Bayes):
    def __init__(self, *args):
        super().__init__(*args)

        self.meanlist = []
        self.covlist = []
        for train in self.trainlist:
            mean, cov = calc_mean_cov(train)
            self.meanlist.append(mean)
            self.covlist.append(cov)
