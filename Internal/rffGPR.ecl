import python3 as python;
IMPORT ML_Core.Types as MTypes;
IMPORT Std.System.Thorlib;
IMPORT $.^ as gpr;
IMPORT gpr.Types as Types;

layout_model2 := MTypes.layout_model2;
NumericField := MTypes.NumericField;
nNodes := Thorlib.nodes();
nodeId := Thorlib.node();
initParams := Types.initParams;


EXPORT rffGPR := MODULE
  // initialize a session
  EXPORT STREAMED DATASET({INTEGER sessID}) init(STREAMED DATASET(initParams) initDat,
                                                  STRING wuid = WORKUNIT) := EMBED(Python:
                                                    GLOBALSCOPE('rffGPR'),PERSIST('global'),activity)
      import numpy as np
      import random as rand
      global get_rffs
      def _get_rffs(X, Win, bin, rff_dim_, sigma_, return_vars):
          N, D = X.shape
          W_ = None
          b_ = None
          if Win is not None:
              W_ = Win 
              b_ = bin 
          else:
              W_ = np.random.normal(loc=0, scale=1, size=(rff_dim_, D))
              b_ = np.random.uniform(0, 2*np.pi, size=rff_dim_)
          B    = np.repeat(b_[:, np.newaxis], N, axis=1)
          norm = 1./ np.sqrt(rff_dim_)
          Z_    = norm * np.sqrt(2) * np.cos(sigma_ * W_ @ X.T + B)
          if return_vars:
              return Z_, W_, b_
          return Z_
      get_rffs = _get_rffs
      sessID = 0
      for rec in initDat:
        nodeID , nNodes = rec
        sessID = nNodes + int(wuid[1:9] + wuid[10:])
      return [(sessID,)]
  ENDEMBED;

  // fit trainig data
  EXPORT DATASET(Layout_model2) fit(INTEGER session, DATASET(NumericField) x,
                                      DATASET(NumericField) y, UNSIGNED4 dim = 10, REAL sig = 1)
                                          := EMBED(Python:GLOBALSCOPE('rffGPR'),PERSIST('global'))
      import numpy as np
      from   scipy.spatial.distance import pdist, cdist, squareform
      from   scipy.linalg import cholesky, cho_solve
      global get_rffs

      rff_dim = dim
      sigma = sig

      # read input data x and y
      dict = {}
      for i in x:
          wi, id, number, data = i
          if id in dict:
              dict[id].append(data)
          else:
              dict[id] = [data]
      X = np.array(list(dict.values()))
      
      yin = []
      for i in y:
          wi, id, number, data = i
          yin.append(data)
      y = np.array(yin)
      # build kernel
      N, D    = X.shape
      Z, W, b = get_rffs(X, None, None, rff_dim, sigma, True)
      sigma_I = sigma * np.eye(N)
      kernel = Z.T @ Z + sigma_I
      lower = True
      L = cholesky(kernel, lower=lower)
      alpha = cho_solve((L, lower), y)
      # return results for prediction
      # the index of each return item is as below
      index = {
                'shapes' : 1,
                'W' : 2,
                'b' : 3,
                'Z' : 4,
                'alpha' : 5
      }
      rst = []
      # return shapes
      rst.append((index['shapes'], sigma, [rff_dim, N, D]))

      # return W 
      for i in range(len(W)):
        for j in range(len(W[0])):
          r = (index['W'], W[i][j], [i, j])
          rst.append(r)

      # return b
      for i in range(len(b)):
        r = (index['b'], b[i], [i])
        rst.append(r)

      # return Z 
      for i in range(len(Z)):
        for j in range(len(Z[0])):
          r = (index['Z'], Z[i][j], [i, j])
          rst.append(r)

      # return alpha
      for i in range(len(alpha)):
        r = (index['alpha'], alpha[i], [i])
        rst.append(r)
      return rst
  ENDEMBED;


  // prediction using data stream to predict the result on each node parallelly.
  EXPORT STREAMED DATASET(NumericField) predict(STREAMED DATASET(Layout_model2) mod,
                                            STREAMED DATASET(NumericField) x, INTEGER session)
                                                := EMBED(Python:GLOBALSCOPE('rffGPR'),
                                                            PERSIST('global'),activity)
      import numpy as np
      global get_rffs
      
      try:
        dict = {}
        i = 0
        ids = []
        prevId = None
        # read input data x
        for rec in x:
            wi, id, number, data = rec
            if id != prevId and prevId != None:
              i += 1
            prevId = id
            
            if i in dict:
                dict[i].append(data)
            else:
                dict[i] = [data]
            ids.append(id)

        X = np.array(list(dict.values()))
        # read model
        if X.shape[0] > 0:
          assert len(X.shape) >= 2, 'bad shape' + str(X.shape)
          index = {
                    'shapes' : 1,
                    'W' : 2,
                    'b' : 3,
                    'Z' : 4,
                    'alpha' : 5
          }
          recs = []
          rff_dim = 0
          N = 0
          D = 0
          sigma = 0.0
          for rec in mod:
            recs.append(rec)
            wi, val, ind = rec
            if wi == index['shapes']:
                rff_dim = ind[0]
                N = ind[1]
                D = ind[2]
                sigma = val
          W = np.zeros((rff_dim, D))
          b = np.zeros(rff_dim)
          Z = np.zeros((rff_dim, N))
          alpha = np.zeros(N)

          for rec in recs:
            wi, val, ind = rec
            if len(ind) > 1:
              r = ind[0]
              c = ind[1]
            else:
              r = ind[0]
            if wi == index['W']:
              W[r][c] = val  
            if wi == index['b']:
              b[r] = val
            if wi == index['Z']:
              Z[r][c] = val  
            if wi == index['alpha']:
              alpha[r] = val
          # predict
          Z_test= get_rffs(X, W, b, rff_dim, sigma, False)
          K_star = Z_test.T @ Z
          y_mean = K_star.dot(alpha)
          rst = []
          for i in range(len(y_mean)):
            rst.append((1, ids[i], 1, y_mean[i]))
          return rst
        else:
          return []
      except:
        import traceback as tb
        exc = tb.format_exc()
        return [(0, 0, 0, -1)]
  ENDEMBED;
END;