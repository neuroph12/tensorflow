import edward as ed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import six
import tensorflow as tf
from edward.models import Categorical, Dirichlet, Empirical, InverseGamma, \
    MultivariateNormalDiag, Normal, ParamMixture

plt.style.use('ggplot')


def build_toy_dataset(N):
  pi = np.array([0.4, 0.3, 0.3])
  mus = [[1, 1], [-1, -1], [-1, 0.5]]
  stds = [[0.1, 0.1], [0.1, 0.1], [0.4, 0.2]]
  x = np.zeros((N, 2), dtype=np.float32)
  C = []
  for n in range(N):
    k = np.argmax(np.random.multinomial(1, pi))
    x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))
    C.append(k)
  return x, np.array(C)


N = 500  # number of data points
K = 3  # number of components
D = 2  # dimensionality of data
ed.set_seed(42)

x_train, C = build_toy_dataset(N)

pi = Dirichlet(tf.ones(K))
mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)
sigmasq = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)
x = ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)}, MultivariateNormalDiag,
                 sample_shape=N)
z = x.cat

T = 500  # number of MCMC samples
qpi = Empirical(tf.Variable(tf.ones([T, K]) / K))
qmu = Empirical(tf.Variable(tf.zeros([T, K, D])))
qsigmasq = Empirical(tf.Variable(tf.ones([T, K, D])))
qz = Empirical(tf.Variable(tf.zeros([T, N], dtype=tf.int32)))

inference = ed.Gibbs({pi: qpi, mu: qmu, sigmasq: qsigmasq, z: qz}, data={x: x_train})
inference.initialize()

sess = ed.get_session()
tf.global_variables_initializer().run()

t_ph = tf.placeholder(tf.int32, [])
running_cluster_means = tf.reduce_mean(qmu.params[:t_ph], 0)

for _ in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)
  t = info_dict['t']
  if t % inference.n_print == 0:
    print("\nInferred cluster means:")
    print(sess.run(running_cluster_means, {t_ph: t - 1}))

# Calculate likelihood for each data point and cluster assignment,
# averaged over many posterior samples. ``x_post`` has shape (N, 100, K, D).
mu_sample = qmu.sample(100)
sigmasq_sample = qsigmasq.sample(100)
x_post = Normal(loc=tf.ones([N, 1, 1, 1]) * mu_sample,
                scale=tf.ones([N, 1, 1, 1]) * tf.sqrt(sigmasq_sample))
x_broadcasted = tf.tile(tf.reshape(x_train, [N, 1, 1, D]), [1, 100, K, 1])

# Sum over latent dimension, then average over posterior samples.
# ``log_liks`` ends up with shape (N, K).
log_liks = x_post.log_prob(x_broadcasted)
log_liks = tf.reduce_sum(log_liks, 3)
log_liks = tf.reduce_mean(log_liks, 1)

# Choose the cluster with the highest likelihood for each data point.
clusters = tf.argmax(log_liks, 1).eval()



fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.scatter(x_train[:, 0], x_train[:, 1], c = C)
ax1.axis([-3, 3, -3, 3])
#ax1.title("Simulated dataset")

ax2.scatter(x_train[:, 0], x_train[:, 1], c=clusters, cmap=cm.bwr)
ax2.axis([-3, 3, -3, 3])
#ax2.title("Predicted cluster assignments")
plt.show()
