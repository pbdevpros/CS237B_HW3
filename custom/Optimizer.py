#! env python

class Optimizer:

    def __init__(self, objective, rate, tolerance, iterations = 100):
        self.objective = objective
        self.rate = rate
        self.iterations = iterations
        self.tolerance = tolerance

    def optimize(self, initial):
        raise Exception("Not implemented!")

class SGD(Optimizer):

    def optimize(self, initial):
        theta = initial
        for i in range(self.iterations):
            # print("iteration: ", i, "theta: ", theta)
            theta_prime = -self.rate * self.objective(theta)
            if abs(theta_prime) <= self.tolerance: 
                break
            theta += theta_prime
        return theta


def main():
    f = lambda x: x ** 2 
    opt = SGD(f, 0.05, 0.0001, 1000000)
    result = opt.optimize(10)
    print(result)

main()