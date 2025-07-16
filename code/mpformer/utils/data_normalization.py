
class ZNorm:
    def __init__(self):
        self.mean = 0.36687853932380676
        self.std = 2.6160225868225098

    def __call__(self, data):
        return (data - self.mean) / self.std