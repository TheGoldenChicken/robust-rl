import numpy as np

X = [3]*40

for i in range(100000):
    iterations = 0
    while True:
        var = 1e-3
        noise = np.random.normal(0, 1e-3, len(X))
        if np.var(noise) <= 3.5e-7: # Adjust this value until the error disappears
            iterations += 1
            continue

        X += noise
        break

    if iterations > 0: print(i)
# print(np.var(noise))
# print(X, iterations)