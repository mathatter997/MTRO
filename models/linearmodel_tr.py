import numpy as np
from numpy import random, linalg
from sympy import Matrix
from scipy.linalg import norm


def sample_with_basis(M):
  weight = np.random.normal(0, 1, len(M))
  v = weight.dot(M)
  # print(v)
  v /= norm(v)
  return v


class LinearModel_TR(object):
  def __init__(self, n_features, learning_rate, n_candidates=0, learning_rate_decay=1.0):
    self.n_features = n_features
    self.learning_rate = learning_rate
    self.n_models = n_candidates + 1
    self.weights = np.zeros((n_features, self.n_models))
    self.learning_rate_decay = learning_rate_decay

  def copy(self):
    copy = LinearModel_TR(n_features=self.n_features, learning_rate=self.learning_rate, n_candidates=self.n_models - 1, learning_rate_decay=1.0)
    copy.weights = self.weights.copy()
    return copy

  def candidate_score(self, features):
    self._last_features = features
    return np.dot(features, self.weights).T

  def score(self, features):
    self._last_features = features
    return np.dot(features, self.weights[:, 0:1])[:, 0]

  def sample_candidates(self, radius):
    assert self.n_models > 1
    random_directions = random.normal(size=(self.n_features, self.n_models - 1))
    random_directions /= linalg.norm(random_directions, axis=0)
    random_radii = random.random(self.n_models - 1) ** (1/self.n_features)
    vectors = radius * (random_directions * random_radii)
    self.weights[:, 1:] = self.weights[:, 0, None] + vectors

  def update_to_mean_winners(self, winners):
    assert self.n_models > 1
    if len(winners) > 0:
      # print 'winners:', winners
      gradient = np.mean(self.weights[:, winners], axis=1) - self.weights[:, 0]
      self.weights[:, 0] += self.learning_rate * gradient
      self.learning_rate *= self.learning_rate_decay

  def update_weights(self, weights):
    self.weights[:, 0] = weights