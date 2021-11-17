IMPORT python3 as python;
IMPORT ML_Core.Types as MTypes;
IMPORT $.^.Types as Types;
NumericField := MTypes.NumericField;

EXPORT DATASET(NumericField) pyGPR(DATASET(NumericField) x, DATASET(NumericField) y) := EMBED(python)
    import numpy as np
    from   sklearn.gaussian_process.kernels import RBF
    from   sklearn.gaussian_process import GaussianProcessRegressor
    dict = {}
    i = 0
    ids = []
    prevId = None
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

    X = np.array(list(dict.values()))[:None]
    kernel  = RBF(1.0, length_scale_bounds="fixed")
    yin = []
    for i in y:
        wi, id, number, data = i
        yin.append(data)
    Y = np.array(yin)[:None]
    clf = GaussianProcessRegressor(kernel=kernel)
    clf = clf.fit(X, Y)
    y_mean= clf.predict(X, return_cov=False)
    score = clf.score(X,Y)
    rst = []
    n = len(y_mean)
    for i in range(n):
      rst.append((1, i+1, 1, y_mean[i]))
    rst.append((1, n+1, 1, score))
    return rst
ENDEMBED;

