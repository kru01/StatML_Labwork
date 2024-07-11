"""
 CSC15004 - 21CNTThuc - Lab assignment 2: Logistic regression
 21127135 - Diep Huu Phuc
"""
import json
import numpy as np
import pandas as pd

from map_feature import map_feature

f_train = 'training_data.txt'
f_config = 'config.json'

class LogRegress:
    def __init__(self, alpha=0.5, lmda=1, iters=10000, verbose=False) -> None:
        self.alpha = alpha
        self.lmda = lmda
        self.iters = iters
        self.verbose = verbose
        self.theta = None

    def sigmoid(self, z): # z = theta.T @ X
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X:np.ndarray, y:np.ndarray, theta:np.ndarray):
        m = y.shape[0]
        h = self.sigmoid(theta[None, :] @ X.T).T
        return 1/m * (-y.T @ np.log(h) - (1-y).T @ np.log(1-h)) \
            + self.lmda/(2*m) * np.sum(theta[1:]**2)

    def compute_gradient(self, X:np.ndarray, y:np.ndarray, theta:np.ndarray):
        m = y.shape[0]
        h = self.sigmoid(theta[None, :] @ X.T)
        loss = h.T - y
        dJ = 1/m * (loss.T @ X) + self.lmda * theta / m
        dJ[0, 0] = 1/m * (loss.T @ X[:, 0]).item()
        return dJ.squeeze()

    def gradient_descent(self, X:np.ndarray, y:np.ndarray):
        theta = np.zeros(X.shape[1])
        print(f'The total of training sample: {len(y)}')
        for i in range(self.iters):
            J = self.compute_cost(X, y, theta)
            dJ = self.compute_gradient(X, y, theta)
            theta = theta - self.alpha * dJ
            if self.verbose:
                mark, cnt = 500, i + 1
                if cnt >= mark and cnt % 100 == 0 \
                    or cnt < mark or cnt == self.iters:
                    print(f'Iter {cnt}, cost = {J}')
        return theta

    def fit(self, X:np.ndarray, y:np.ndarray):
        self.theta = self.gradient_descent(X, y)

    def predict(self, X:np.ndarray):
        return np.array([0 if self.sigmoid(self.theta @ Xi)
                         < 0.5 else 1 for Xi in X])[:, None]
    
    def evaluate(self, y:np.ndarray, y_pred:np.ndarray):
        classes = np.unique(y)
        cmat = np.zeros((len(classes), len(classes)))
            # TP FP
            # FN TN
        for i in range(len(classes)):
            for j in range(len(classes)):
                cmat[i, j] = np.sum((y == classes[i])
                                        & (y_pred == classes[j]))

        acc = (cmat[0, 0] + cmat[1, 1]) / np.sum(cmat)
        prec = cmat[0, 0] / (cmat[0, 0] + cmat[0, 1])
        rec = cmat[0, 0] / (cmat[0, 0] + cmat[1, 0])
        f1sc = 2 * prec * rec / (prec + rec)
        return acc, prec, rec, f1sc

def main(df:pd.DataFrame, configs:dict):
    print(df.head(), '\n', df.describe())

    X = df.iloc[:, :-1].to_numpy()
    X = map_feature(X[:, 0], X[:, 1])
    y = df.iloc[:, -1].to_numpy()[:, None]
    print(f'X: {X.shape}, y: {y.shape}')

    model = LogRegress(alpha=configs["Alpha"], lmda=configs["Lambda"],
                       iters=configs["NumIter"], verbose=False)
    model.fit(X, y)
    with open('model.json', 'w') as f:
        json.dump({'theta: ': model.theta.tolist()}, f)
    print('-- Model successfully saved to model.json')

    samples = np.array([[0.051267, 0.69956]])
    samples = map_feature(samples[:, 0], samples[:, 1])
    preds = model.predict(samples)
    print(f'-- Testing samples: {samples.shape}\n{preds}')
    print(f'-- Evaluation on training dataset')

    y_pred = model.predict(X)
    scores = model.evaluate(y, y_pred)
    print(f'(Accuracy, Precision, Recall, F1-score)\n{scores}')

    with open('classification_report.json', 'w') as f:
        result = {
            'Accuracy': scores[0],
            'Precision': scores[1],
            'Recall': scores[2],
            'F1score': scores[3]
        }
        json.dump(result, f)
    print('-- Scores successfully saved to classification_report.json')

if __name__=="__main__":
    with open(f_config,) as f: configs = json.load(f)
    df = pd.read_csv(f_train, header=None)
    main(df, configs)