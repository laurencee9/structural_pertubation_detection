
import networkx as nx
import numpy as np

from sklearn.linear_model import LinearRegression


class TimeGrangerCausality():
    """Infers Granger causality."""

    @staticmethod
    def split_data(x, lag):

        T = len(x)
        inputs = np.zeros([T - lag - 1, lag])
        targets = np.zeros(T - lag - 1)

        for t in range(T - lag - 1):
            inputs[t, :] = x[t : lag + t]
            targets[t] = x[t + lag] #-> First error in netrd implementation

        return inputs, targets


    def fit(self, X, lag=1):

        N = X.shape[0]
        W_pred = np.zeros([N, N])

        for i in range(N):
            xi, yi = TimeGrangerCausality.split_data(X[i, :], lag)

            for j in range(N):
                xj, yj = TimeGrangerCausality.split_data(X[j, :], lag)
                xij = np.concatenate([xi, xj], axis=-1)
                reg1 = LinearRegression().fit(xi, yi)
                reg2 = LinearRegression().fit(xij, yi)
                err1 = yi - reg1.predict(xi)
                err2 = yi - reg2.predict(xij)
                
                std_i = np.std(err1)
                std_ij = np.std(err2)
                if std_i==0:
                    W_pred[j,i] = -99999
                elif std_ij==0:
                    W_pred[j,i] = 999999999
                else:
                    W_pred[j,i] = np.log(std_i) - np.log(std_ij)

        return nx.from_numpy_array(W_pred, create_using=nx.DiGraph())





