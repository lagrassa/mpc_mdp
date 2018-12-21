import mdptoolbox as mdpt
import numpy as np
P = np.array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]]) #A,S,S so 2 states, 2 actions
R = np.array([[5, 10], [-1, 2]]) #(S,A)
discount = 0.9
vi = mdpt.mdp.ValueIteration(P, R, discount)
vi.run()
value_function = vi.V
print(value_function)

