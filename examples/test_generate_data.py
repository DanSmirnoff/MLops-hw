import numpy as np
import matplotlib.pyplot as plt


def generate_spirals(n_points=500, noise=0.5, revolutions=4):

    np.random.seed(42)

    theta1 = np.sqrt(np.random.rand(n_points)) * revolutions * np.pi
    r1 = theta1 + noise * np.random.randn(n_points)
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)

    theta2 = np.sqrt(np.random.rand(n_points)) * revolutions * np.pi
    r2 = theta2 + noise * np.random.randn(n_points)
    x2 = r2 * np.cos(theta2 + np.pi)
    y2 = r2 * np.sin(theta2 + np.pi)


    X = np.vstack([np.column_stack([x1, y1]), 
                   np.column_stack([x2, y2])]).tolist()
    y = np.hstack([np.zeros(n_points), np.ones(n_points)]).tolist()

    return X, y


if __name__ == "__main__":

    X, y = generate_spirals(n_points=500, noise=0.5, revolutions=4)
    X, y = np.array(X), np.array(y)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', label='Класс 0', alpha=0.7)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='Класс 1', alpha=0.7)
    plt.title('Две закручивающиеся спирали')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

    print(f"Размерность данных: {X.shape}")
