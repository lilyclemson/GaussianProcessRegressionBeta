IMPORT python3 as python;
IMPORT ML_Core.Types as Types;

// Generate testing data
EXPORT M_dataGen(INTEGER n, INTEGER n_train) := MODULE

EXPORT l := RECORD
 set of real x;
end;

SHARED SET OF INTEGER randChoice(INTEGER n, INTEGER n_train) := EMBED(python)
    import numpy as np
    inds    = np.random.choice(n, size=n_train, replace=False)
    if 0 not in inds:
        inds[0] = 0
    if n-1 not in inds:
        inds[n_train-1] = n-1
    assert(np.unique(inds).size == inds.size)
    return inds.tolist()
ENDEMBED;

SHARED seed := randChoice(n, n_train);

EXPORT toNF(set of REAL input) := PROJECT(DATASET(input, {REAL x}),
                                         TRANSFORM(Types.NumericField, SELF.wi := 1,
                                                                       SELF.id := COUNTER,
                                                                       SELF.number := 1,
                                                                       SELF.value := LEFT.x));

EXPORT set of real generateXData(INTEGER n) := EMBED(python)
  import numpy as np
  # Create 100 random x values
  X = np.linspace(0, 50, n)
  return X.tolist()
ENDEMBED;

EXPORT x :=  generateXData(n);


EXPORT set of real generateYData(set of real x_data) := EMBED(python)
  import numpy as np
  from   sklearn.gaussian_process.kernels import RBF
  kernel = RBF(1.0, length_scale_bounds="fixed");
  X = np.array(x_data)[:,None]
  n = len(x_data)
  mean = np.zeros(n)
  cov = kernel(X.reshape(n, -1))
  y = np.random.multivariate_normal(mean, cov)
  return y.tolist()
ENDEMBED;

EXPORT y := generateYData(x);

SHARED set of real getX(SET OF INTEGER seed, set of real xin) := EMBED(python)
    import numpy as np
    sd = []
    for i in seed:
      sd.append(i)
    return np.array(xin)[seed].tolist()
ENDEMBED;

SHARED SET OF REAL getY(SET OF INTEGER seed, set of real yin ) := EMBED(python)
    import numpy as np
    sd = []
    for i in seed:
      sd.append(i)
    return np.array(yin)[seed].tolist()
ENDEMBED;
// EXPORT x_all := toNF(x);
// EXPORT y_all := toNF(y);
EXPORT X_train := toNF(x);
EXPORT Y_train := toNF(y);
EXPORT X_test := toNF(getX(seed, x));
EXPORT Y_test := toNF(getY(seed, y));

END;