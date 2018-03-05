class Metrics(object):
        def __init__(self):
                self.min = None
                self.max = None
                self.dev = None
                self.mean = None
                self.med = None
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
