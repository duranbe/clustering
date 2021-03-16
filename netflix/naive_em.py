"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture
import scipy
from scipy import stats
import collections


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """

    n, d = X.shape
    K, _ = mixture.mu.shape

    log_likelihood = 0.0

    post = np.zeros(shape=(n, K))

    for i in range(n): #For each point

        prb = np.zeros(shape=K)

        for j in range(0, K):

            x = X[i]
            mu = mixture.mu[j]
            var = mixture.var[j]
            p = mixture.p[j]

            first = 1 / ((2 * np.pi * var) ** (d / 2))

            second = np.exp(-1 * (np.linalg.norm(x - mu) ** 2) / (2 * var))

            prb[j] = p * first * second

        log_likelihood += np.log(np.sum(prb))

        prb = prb / np.sum(prb)

        post[i] = prb

    return (post, log_likelihood)


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape

    new_mu = []
    new_p = []
    new_var = []

    print("Count :", dict(collections.Counter(np.argmax(post, axis=1))))

    for i in range(K):

        nj = np.sum(post[:, i])

        pj = nj / n

        muj = (1 / nj) * np.sum(post[:, i].reshape(250, 1) * X, axis=0)

        norm = np.linalg.norm(X - muj, axis=1)

        varj = (1 / (nj * d)) * np.sum(post[:, i] * (norm ** 2))

        new_mu.append(muj)
        new_p.append(pj)
        new_var.append(varj)

    return GaussianMixture(
        mu=np.array(new_mu), var=np.array(new_var), p=np.array(new_p)
    )


def run(
    X: np.ndarray, mixture: GaussianMixture, post: np.ndarray
) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment

    """

    old_likelihood = 1000000

    while True:

        post, log_likelihood = estep(X, mixture)
        mixture = mstep(X, post)

        print("Log Likelihood :", log_likelihood)

        if np.abs(old_likelihood - log_likelihood) < (10 ** -6):
            return (mixture, post, log_likelihood)
        else:
            old_likelihood = log_likelihood


if __name__ == "__main__":
    pass
