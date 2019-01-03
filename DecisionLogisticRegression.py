import numpy as np
import pandas as pd
import pdb
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



class DecisionLogisticRegression(DecisionTreeClassifier):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def fit(self,**kwargs):
        super().fit(**kwargs)
        self.dt = super(DecisionLogisticRegression,self)
        X = pd.DataFrame( kwargs['X'] )
        X_cols = X.columns.tolist()
        tmp = X.copy().reset_index(drop=True)
        tmp['__y'] = kwargs['y']
        tmp['__leaf'] = self.dt.apply(X)
        leaves = tmp['__leaf'].unique().tolist()
        leaves.sort()

        self.leaves = leaves
        self.lrs = dict()
        self.clfs = {leaf:self.dt for leaf in leaves}
        for leaf in leaves:
            _tmp  = tmp[tmp['__leaf'] == leaf]
            _X = _tmp[X_cols]
            _y = _tmp['__y']
            self.lrs[leaf] = LogisticRegression(random_state=0)
            if len( _y.unique().tolist() ) > 1:
                self.lrs[leaf].fit(_X,_y)
                if self.lrs[leaf].score(_X,_y) > accuracy_score(y_pred=self.dt.predict(_X),y_true=_y):
                    self.clfs[leaf] = self.lrs[leaf]
        return self

    def predict_proba(self,**kwargs):
        X = pd.DataFrame( kwargs['X'] )
        X_cols = X.columns.tolist()
        tmp = X.copy().reset_index()
        ori_index = tmp.index.tolist()
        # probs_df = pd.DataFrame(self.dt.predict_proba(tmp[X_cols]),
        #                         columns=self.classes_,index=tmp.index)
        # probs_cols = probs_df.columns.tolist()
        tmp['__leaf'] = self.apply(X)

        dfs = []
        for leaf in self.leaves:
            _tmp = tmp[tmp['__leaf'] == leaf]
            _X = _tmp[X_cols]
            try:
                _probs_df = pd.DataFrame(self.clfs[leaf].predict_proba(_X),
                                        columns=self.clfs[leaf].classes_,
                                        index=_tmp.index)
            except:
                _probs_df = pd.DataFrame(self.clfs[leaf].predict_proba(_X),
                                        columns=self.classes_,
                                        index=_tmp.index)
            dfs.append(_probs_df)
        return pd.concat(dfs).fillna(0).loc[ori_index]

    def predict_log_proba(self,**kwargs):
        X = pd.DataFrame( kwargs['X'] )
        X_cols = X.columns.tolist()
        tmp = X.copy().reset_index()
        ori_index = tmp.index.tolist()
        # probs_df = pd.DataFrame(self.dt.predict_log_proba(tmp[X_cols]),
        #                         columns=self.classes_,index=tmp.index)
        # probs_cols = probs_cols.columns.tolist()
        tmp['__leaf'] = self.apply(X)

        dfs = []
        for leaf in self.leaves:
            _tmp = tmp[tmp['__leaf'] == leaf]
            _X = _tmp[X_cols]
            try:
                _probs_df = pd.DataFrame(self.clfs[leaf].predict_proba(_X),
                                        columns=self.clfs[leaf].classes_,
                                        index=_tmp.index)
            except:
                _probs_df = pd.DataFrame(self.clfs[leaf].predict_proba(_X),
                                        columns=self.classes_,
                                        index=_tmp.index)
            dfs.append(_probs_df)

        return pd.concat(dfs).fillna(0).loc[ori_index]

    def predict(self,**kwargs):
        probs_df = self.predict_proba(**kwargs)
        return probs_df.apply(lambda row: row.idxmax(),axis=1)






if __name__ == '__main__':
    X = pd.DataFrame(np.random.randn(1000, 50),columns=['feature_{}'.format(i) for i in range(50)])
    y = ['A'] * 300 + ['B'] * 300 + ['C'] * 400

    clf2 = DecisionLogisticRegression(random_state=0,criterion='entropy',**{'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 0.01})
    clf2.fit(X=X,y=y)
    print( clf2.predict_proba(X=X) )
    print( clf2.predict_log_proba(X=X) )
    print( clf2.predict(X=X) )
