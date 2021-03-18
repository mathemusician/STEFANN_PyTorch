import torch
import torch.nn as nn



class LossMeter(object):
    def __init__(self):
        super().__init__()
        self.reset()


    def reset(self):
        self._value=0
        self.sum=0
        self.count=0


    def update(self,loss,count):
        self.count+=count
        self.sum+=count*loss
        self._value=self.sum/self.count


    @property
    def value(self):
        return self._value

