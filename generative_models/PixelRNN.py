import numpy as np
time_steps = 100
input_feature = 32
output_feature = 64
inputs = np.random.random((time_steps, input_feature))
state_t = np.zeros((output_feature,))
W = np.random.random((output_feature, input_feature))
U = np.random.random((output_feature, output_feature))
b = np.random.random(output_feature)

successive_output = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_output.append(output_t)
    state_t = output_t

final_output_sequence = np.stack(successive_output, axis=0)
