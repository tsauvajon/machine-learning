from numpy import *

def compute_errors_for_given_points():
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))

def step_gradient(b_current, m_current, points, learning_rate):
    # gradient descent
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += (y - (m_current * x + b_current)) * (-2 / N)
        m_gradient += (y - (m_current * x + b_current)) * (-2 / N) * x
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations):
    b = initial_b
    m = initial_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run():
    points = genfromtxt('data.csv', delimiter=',')
    # hyper parameter : à quelle vitesse le modèle apprend
    learning_rate = 0.0001

    # y = mx + b
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print(b)
    print(m)

if __name__ == '__main__':
    run()
