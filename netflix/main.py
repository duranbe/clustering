import numpy as np
import kmeans
import common
import naive_em
import em

K = 3
np.random.seed(0)
X = np.loadtxt("toy_data.txt")

GM = common.GaussianMixture(
    mu=np.random.rand(K, 2), var=np.random.rand(K), p=np.random.rand(K)
)

GM,post,cost = kmeans.run(X,GM,None)

rep = naive_em.run(X, GM, None)

common.plot(X, rep[0], rep[1], f"Toy Dataset with K = {K}")
