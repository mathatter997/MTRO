# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import utils.rankings as rnk
from models.linearmodel_tr import LinearModel_TR
from algorithms.basiconlineranker import BasicOnlineRanker
from multileaving.ProbabilisticMultileave import ProbabilisticMultileave
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.pipeline import Pipeline 

# Probabilistic Interleaving Dueling Bandit Gradient Descent
class P_MTRO(BasicOnlineRanker):
  def __init__(self, PM_n_samples, PM_tau, learning_rate, learning_rate_decay,
              radius, radius_max, p, alpha, beta_1, beta_2, candidates, *args, **kargs):
    super(P_MTRO, self).__init__(*args, **kargs)
    n_candidates = max(self.n_features, candidates)
    self.model = LinearModel_TR(n_features = self.n_features,
                             learning_rate = learning_rate,
                             n_candidates = n_candidates,
                             learning_rate_decay = learning_rate_decay)
    self.compare_model = LinearModel_TR(n_features = self.n_features,
                             learning_rate = learning_rate,
                             n_candidates = 1,
                             learning_rate_decay = learning_rate_decay)
    self.multileaving = ProbabilisticMultileave(
                             n_samples = PM_n_samples,
                             tau = PM_tau,
                             n_results=self.n_results)
    self.local_estimator = Pipeline([('poly', PolynomialFeatures(1)), ('regression', linear_model.LinearRegression())]) 
    self.radius = radius 
    self.radius_max = radius_max
    self.alpha = alpha
    self.p = p
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.grad_norm = 0

  @staticmethod
  def default_parameters():
    parent_parameters =  BasicOnlineRanker.default_parameters()
    parent_parameters.update({
      # main optimization parameters
      'learning_rate': 0.01,
      'learning_rate_decay': 1.0,
      'PM_n_samples': 1000,
      'PM_tau': 3.0,
      'radius': 0.2,
      'radius_max': 1,
      'p': 0.6,
      # adaptive model construction
      'alpha': 0.99,
      'beta_1': 1.01,
      'beta_2': 0.999,
      'candidates': 91,
      })
    return parent_parameters

  def _create_train_ranking(self, query_id, query_feat, inverted):
    assert inverted==False
    self.model.sample_candidates(self.radius * self.w ** (self.j - 1))
    scores = self.model.candidate_score(query_feat)
    inverted_rankings = rnk.rank_single_query(scores,
                                              inverted=True,
                                              n_results=None)
    multileaved_list = self.multileaving.make_multileaving(inverted_rankings)
    return multileaved_list

  def update_to_interaction(self, clicks):
    if not np.any(clicks):
      self.radius = self.radius * self.beta_2
    else:
      X = self.model.weights.T 
      y = self.multileaving.infer_preferences(clicks)[0, :]
      self.local_estimator.fit(X, y)
      r_2 = self.local_estimator.score(X,y)
      gradient = self.get_gradient(X[0])
      self.grad_norm = np.linalg.norm(gradient)
      alpha = self.radius
      if self.grad_norm:
        alpha = alpha / self.grad_norm
      if r_2 > self.p:
        X_new = X[0] - alpha * gradient 
        self.model.update_weights(X_new)
        self.radius * self.alpha
      else:
        self.radius * self.beta_1
      self.radius = min(self.radius, self.radius_max)
  
  def get_test_rankings(self, features,
                        query_ranges, inverted=True):
    scores = self.model.score(features)
    return rnk.rank_multiple_queries(
                      scores,
                      query_ranges,
                      inverted=inverted,
                      n_results=self.n_results)
  
  def get_gradient(self, point):
    powers = self.local_estimator['poly'].powers_.copy()
    coef = self.local_estimator['regression'].coef_.copy()
    D = powers.shape[1]
    grad = np.zeros(D)
    for d in range(D):
        p = powers[:,d]
        p = np.multiply(p, coef)
        pow = powers[:, d].copy() 
        powers[:, d] = (powers[:, d] - 1).clip(min=0)
        grad[d] = p.dot(np.prod(point ** powers, axis=1))
        powers[:, d] = pow
    return grad

  def get_hessian(self, point):
    powers = self.local_estimator['poly'].powers_.copy()
    coef = self.local_estimator['regression'].coef_.copy()
    D = powers.shape[1]
    hess = np.zeros(shape=(D, D))
    for i in range(D):
        for j in range(i, D):
            pi = powers[:,i]
            p = np.multiply(pi, coef)
            pow_i = powers[:, i].copy()
            if i != j:
                pow_j = powers[:, j].copy()
            powers[:, i] -= 1
            pj = powers[:,j]
            p = np.multiply(p, pj)
            powers[:, j] -= 1
            powers[:, i] = powers[:, i].clip(min=0)
            if i != j:
                powers[:, j] = powers[:, j].clip(min=0)
            hess[i][j] = hess[j][i] = p.dot(np.prod(point ** powers, axis=1))
            powers[:, i] = pow_i
            if i != j:
                powers[:, j] = pow_j  
    return hess

  # Steihaug conjugate gradient
  def minimize(self, point, epsilon=0.01):
    g = self.get_gradient(point)
    B = self.get_hessian(point)
    # step 1
    p = point
    d = -1 * g
    for _ in range(g.size):
        # step 2
        Bd = np.matmul(B, d)
        dBd = np.dot(d, Bd)
        if dBd <= 0:
            v = p - point
            roots = np.roots([np.dot(d, d), 2 * np.dot(v, d), np.dot(v, v) - self.radius ** 2])
            tau = np.max(roots)
            return p + tau * d
        # step 3
        dd = np.dot(d, d)
        alpha = dd / dBd
        p_new = p + alpha * d
        if np.linalg.norm(p_new - point) >= self.radius:
            v = p - point
            roots = np.roots([dd, 2 * np.dot(v, d), np.dot(v, v) - self.radius ** 2])
            tau = np.max(roots)
            return p + tau * d
        d_new = d - alpha * Bd
        # step 4
        if np.linalg.norm(d_new)/np.linalg.norm(g) < epsilon:
            return p_new
        # step 5
        dd_new = np.dot(d_new, d_new)
        beta = dd_new / dd
        d_new = d_new + beta * d
        p = p_new
        d = d_new
    return p

