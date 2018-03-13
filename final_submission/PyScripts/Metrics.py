class Metrics(object):
        def __init__(self, in_min=None, in_max=None, in_dev=None, in_mean=None, in_med=None):
                self.min = in_min
                self.max = in_max
                self.dev = in_dev
                self.mean = in_mean
                self.med = in_med
        def __repr__(self):
                return self.__str__()
        def __str__(self):
                return str({
                        'max': self.max,
                        'min': self.min,
                        'dev': self.dev,
                        'mean': self.mean, 
                        'med': self.med,
                })
        def getMin(self):
                return self.min
        def getMax(self):
                return self.max
        def getMean(self):
                return self.mean
        def getDev(self):
                return self.dev
        def getMed(self):
                return self.med
