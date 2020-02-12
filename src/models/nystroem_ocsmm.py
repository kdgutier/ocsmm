import warnings

import numpy as np
from scipy.linalg import svd

from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDClassifier

from sklearn.utils import check_random_state, check_X_y
from sklearn.utils.validation import _num_samples, check_is_fitted
from sklearn.utils.multiclass import _check_partial_fit_first_call

from sklearn.utils import compute_class_weight

from models.kernels import kernel, gramMeasure, cross_gramMeasure, measureNormSquare


class NystroemSet():
  """
    Approximate a kernel map using a subset of the training data.
    Constructs an approximate feature map for a rbf
    using a subset of the data as basis.

    Parameters
    ----------
    gamma : float, default=None
        Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
        and sigmoid kernels. Interpretation of the default value is left to
        the kernel; see the documentation for sklearn.metrics.pairwise.
        Ignored by other kernels.
    n_components : int
        Number of features to construct.
        How many data points will be used to construct the mapping.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Subset of training points used to construct the feature map.
    component_indices_ : array, shape (n_components)
        Indices of ``components_`` in the training set.
    normalization_ : array, shape (n_components, n_components)
        Normalization matrix needed for embedding.
        Square root of the kernel matrix on ``components_``.
  """
    
  def __init__(self, gamma, n_components=100, random_state=None):
    self.gamma = gamma
    self.n_components = n_components
    self.random_state = random_state

  def nystroem_fit(self, S):
    """Fit estimator to data.
    Samples a subset of training points, computes kernel
    on these and computes normalization matrix.
    Parameters
    ----------
    X : array-like, shape=(n_samples, n_feature)
        Training data.
    """
    rnd = check_random_state(self.random_state)
    n_samples = S.shape[0]

    # get basis vectors
    if self.n_components > n_samples:
      n_components = n_samples
      warnings.warn("n_components > n_samples. This is not possible.\n"
                    "n_components was set to n_samples, which results"
                    " in inefficient evaluation of the full kernel.")

    else:
      n_components = self.n_components
    n_components = min(n_samples, n_components)
    inds = rnd.permutation(n_samples)
    basis_inds = inds[:n_components]
    basis = S[basis_inds]
    
    basis_kernel = cross_gramMeasure(basis, basis, self.gamma)

    # sqrt of kernel matrix on basis vectors
    U, D, V = svd(basis_kernel)
    D = np.maximum(D, 1e-12)
    self.normalization_ = np.dot(U / np.sqrt(D), V)
    self.components_ = basis
    self.component_indices_ = inds
    return self

  def nystroem_transform(self, S):
    """Apply feature map to S.
    Computes an approximate feature map using the kernel
    between some training sets and S.
    Parameters
    ----------
    S : list-like, list of training data matrices with shape (n_samples, n_features)

    Returns
    -------
    S_transformed : Transformed sets.
    """
    check_is_fitted(self, 'components_')
    embedded = cross_gramMeasure(S, self.components_, self.gamma)        
    return np.dot(embedded, self.normalization_.T)


class NystroemOneClassSMM(OneClassSVM, NystroemSet): 
  """Nystroem One Class Support Measure Machines (OCSMM).
  Input
  ----------
  S : list-like, list of training data matrices with shape (n_samples, n_features)
  C : regularization parameter
  gamma: float, default=None. Gamma parameter for the RBF.
  References
  ----------
  .. [MuandetScholkpf2013] One-Class Support Measure Machines for 
      Group Anomaly Detection, 2013, Proceedings of the Twenty-Ninth 
      Conference on Uncertainty in Artificial Intelligence
  .. [LuWang2016] Large Scale Online Kernel Learning.
      Journal of Machine Learning Research 2016.
  """
  def __init__(self, C, gamma, n_components=100, random_state=None):
    NystroemSet.__init__(self, gamma=gamma, n_components=n_components,
                         random_state=random_state)
    OneClassSVM.__init__(self, nu=C, gamma=gamma, kernel='linear')

  def fit(self, Strain):
    Strain = np.array(Strain)

    super().nystroem_fit(Strain)
    NStrain = super().nystroem_transform(Strain)

    super().fit(X=NStrain)
    return self
    
  def outlier_score(self, Stest):
    Stest = np.array(Stest)
    NStest = super().nystroem_transform(Stest)

    outlier_score = -(super().decision_function(NStest) + self.offset_)
    return outlier_score

  def predict(self, Stest, thr):
    Stest = np.array(Stest)
    NStest = super().nystroem_transform(Stest)
    y_hat = super().predict(NStest)
    return y_hat


class OnlineNystroemOneClassSMM(SGDClassifier, NystroemSet): 
  def __init__(self, C, gamma, learning_rate="optimal", class_weight=None, n_components=100, random_state=None):
    self.C = C
    self.gamma = gamma
    self.learning_rate = learning_rate
    self.loss='hinge'
    self.penalty='l2'
    NystroemSet.__init__(self, gamma=gamma, n_components=n_components,
                         random_state=random_state)
    SGDClassifier.__init__(self, loss=self.loss, penalty=self.penalty,
                           alpha=1/C, warm_start=True, class_weight=class_weight)

  def fit(self, Strain, sample_weight=None, coef_init=None, intercept_init=None):
    Strain = np.array(Strain)

    if not hasattr(self, 'components_'):
      super().nystroem_fit(Strain)

    if self.warm_start and hasattr(self, "coef_"):
      if coef_init is None:
        coef_init = self.coef_
      if intercept_init is None:
        intercept_init = self.intercept_
      else:
        self.coef_ = None
        self.intercept_ = None

      if self.average > 0:
        self.standard_coef_ = self.coef_
        self.standard_intercept_ = self.intercept_
        self.average_coef_ = None
        self.average_intercept_ = None
      
      super().nystroem_fit(Strain)

    NStrain = super().nystroem_transform(Strain)
    ytrain = np.ones(_num_samples(Strain))
    
    # Clear iteration count for multiple call to fit.
    self.t_ = 1.0

    classes = np.array([-1, 1])

    self._partial_fit(X=NStrain, y=ytrain, alpha=self.alpha, C=1.0,
                      loss=self.loss, learning_rate=self.learning_rate, max_iter=1000,
                      classes=classes, sample_weight=None,
                      coef_init=coef_init, intercept_init=intercept_init)
    return self
  
  def _partial_fit(self, X, y, alpha, C,
                   loss, learning_rate, max_iter,
                   classes, sample_weight=None,
                   coef_init=None, intercept_init=None):
    X, y = check_X_y(X, y, 'csr', dtype=np.float64, order="C",
                     accept_large_sparse=False)

    n_samples, n_features = X.shape

    _check_partial_fit_first_call(self, classes)

    n_classes = 2

    # Allocate datastructures from input arguments
    self._expanded_class_weight = compute_class_weight(self.class_weight,
                                                       self.classes_, y)
    if sample_weight is None:
      sample_weight = np.ones(n_samples, dtype=np.float64)

    if getattr(self, "coef_", None) is None or coef_init is not None:
      self._allocate_parameter_mem(n_classes, n_features,
                                   coef_init, intercept_init)
    elif n_features != self.coef_.shape[-1]:
      raise ValueError("Number of features %d does not match previous "
                       "data %d." % (n_features, self.coef_.shape[-1]))

    self.loss_function_ = self._get_loss_function(loss)
    if not hasattr(self, "t_"):
      self.t_ = 1.0
      
    self._fit_binary(X, y, alpha=alpha, C=C,
                     learning_rate=learning_rate,
                     sample_weight=sample_weight,
                     max_iter=max_iter)
    return  self

  def outlier_score(self, Stest):
    Stest = np.array(Stest)
    NStest = super().nystroem_transform(Stest)

    outlier_score = -(super().decision_function(NStest))
    return outlier_score

  def predict(self, Stest):
    Stest = np.array(Stest)
    NStest = super().nystroem_transform(Stest)
    y_hat = super().predict(NStest)
    return y_hat
