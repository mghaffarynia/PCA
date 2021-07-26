

import pandas as pd
import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import math


def read_csv_input(filename):
  df = pd.read_csv(filename, header = None).to_numpy()
  y = df[:, [-1]]
  X = df[:, range(df.shape[1]-1)]
  return X, y

def opt(X, y, c):
  m, n = X.shape
  P_top = np.concatenate((np.identity(n), np.zeros((n, m+1))), axis=1)
  P = matrix(np.concatenate((P_top, np.zeros((m+1, n+m+1))), axis=0))
  np_q = np.concatenate((np.zeros((n+1, 1)), np.ones((m, 1))*c), axis=0)
  q = matrix(np_q)
  G_top_left = (-1)*y*np.concatenate((X, np.ones((m, 1))), axis=1)
  G_top = np.concatenate((G_top_left, (-1)*np.identity(m)), axis=1)
  G_down = np.concatenate((np.zeros((m, n+1)), np.identity(m)*-1), axis=1)
  np_G = np.concatenate((G_top, G_down), axis = 0)
  G = matrix(np_G)
  np_h = np.concatenate((np.ones((m, 1))*(-1), np.zeros((m, 1))), axis=0)
  h = matrix(np_h)
  solvers.options['show_progress'] = False
  sol = np.array(solvers.qp(P, q, G, h)['x'])
  return sol[:n, :], sol[n][0]

def normalize(X):
  means = np.mean(X, axis=0)
  return (X-means), means

def get_eigen(W):
  cov = np.dot(W, W.T)
  lambdas, v = np.linalg.eig(cov)
  return lambdas, v

def predict(w, b, X):
  predicts = np.where(np.dot(X, w) + b < 0, -1, 1)
  return predicts

def accuracy(prediction, y):
  return np.sum(np.where(prediction*y <= 0, 0, 1))/y.shape[0]*100

def print_errors(accuracies, c_list):
  print("\t      c:", end='')
  for c in c_list:
    print(f"{c:6}", end=' ')
  print()
  for k, accs in enumerate(accuracies):
    print(f"\tk:{k+1}", end='\t')
    for acc in accs:
      print(f"{100-acc:6.2f}", end=' ')
    print()
  print()

def pi_j(lambdas, v, k):
  print(f"v shape is : {v.shape}")
  sorted_indices = np.argsort(lambdas)
  sq = np.multiply(v[:, sorted_indices[::-1][:k]], v[:, sorted_indices[::-1][:k]])
  norm = (1/k)*(np.sum(sq, axis=1, keepdims=True))
  print(f"The shape of pi_j matrix is : {norm.shape}")
  s = int(k * math.log(k))
  samples = np.random.choice(norm.shape[1], s)

  return norm, samples

def main():
  # np.set_printoptions(linewidth=np.inf)
  X, y = read_csv_input("madelon.data")
  m, n = X.shape
  X_train, y_train = X[:int(0.6*m), :], y[:int(0.6*m), :]
  X_val, y_val = X[int(0.6*m):int(0.9*m) , :], y[int(0.6*m):int(0.9*m) , :]
  X_test, y_test = X[int(0.9*m): , :], y[int(0.9*m): , :]

  # X_train, y_train = read_csv_input("sonar_train.csv")
  # X_val, y_val = read_csv_input("sonar_valid.csv")
  # X_test, y_test = read_csv_input("sonar_test.csv")
  # m, n = X_train.shape

  X_train_normalized, means = normalize(X_train)
  X_val_normalized = (X_val-means)
  X_test_normalized = (X_test-means)

  lambdas, v = get_eigen(X_train_normalized.T)
  sorted_indices = np.argsort(lambdas)
  print("Top six eigenvalues are:")
  print("\t", lambdas[sorted_indices[::-1][:6]])
  accuracies_train, accuracies_val, accuracies_test = [], [], []
  K, c_list = 100, [0.001, 0.01, 0.1, 1, 1e12]
  projections_train = np.dot(X_train_normalized, v[:, sorted_indices[::-1][:K]])
  projections_val = np.dot(X_val_normalized, v[:, sorted_indices[::-1][:K]])
  projections_test = np.dot(X_test_normalized, v[:, sorted_indices[::-1][:K]])

  for k in range(K):
    accs_train, accs_val, accs_test = [], [], []
    for c in c_list:
      w, b = opt(projections_train[:, :k+1], y_train, c)
      predictions_train = predict(w, b, projections_train[:, :k+1])
      accs_train.append(accuracy(predictions_train, y_train))
      predictions_val = predict(w, b, projections_val[:, :k+1])
      accs_val.append(accuracy(predictions_val, y_val))
      predictions_test = predict(w, b, projections_test[:, :k+1])
      accs_test.append(accuracy(predictions_test, y_test))
    accuracies_train.append(accs_train)
    accuracies_val.append(accs_val)
    accuracies_test.append(accs_test)

  print("Train errors:")
  print_errors(accuracies_train, c_list)
  print("Validation errors:")
  print_errors(accuracies_val, c_list)
  print("Test errors:")
  print_errors(accuracies_test, c_list)

main()