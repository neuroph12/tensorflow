import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Feed_fwd:
    def __init__(in_units = 2, hidden_units = [5], out_units = 2):
        u_list = [in_units] + hidden_units + [out_units]
        self.W = []
        self.b = []
        for idx in range(len(u_lists) - 1):
            self.W[idx] = tf.Variable(tf.zeros([u_list[idx], u_list[idx + 1]]))
            self.b[idx] = tf.Variables(tf.zeros([u_list[idx + 1]]))

    def inference(x):
        h = x
        for idx in range(len(self.W) - 1):
            h = tf.nn.relu(tf.matmul(h, self.W[idx]) + self.b[idx])
        y = tf.matmul(h, self.W[-1]) + self.b[-1]

    def loss(y, t):
