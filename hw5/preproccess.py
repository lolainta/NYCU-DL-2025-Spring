from collections import deque
import cv2
import numpy as np


class DummyPreprocessor:
    def __init__(self):
        pass

    def preprocess(self, obs):
        return obs

    def reset(self, obs):
        return obs

    def step(self, obs):
        return obs
