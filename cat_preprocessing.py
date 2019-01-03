import pandas as pd
import numpy as np
import pdb
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,LabelBinarizer,StandardScaler
from collections import defaultdict

class MultiColumnsLabelEncoder(LabelEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.les = defaultdict(lambda : LabelEncoder(**kwargs))


    def fit(self,**kwargs):
        cat_X_df = pd.DataFrame(kwargs['y'])
        cat_X_df.apply(lambda x: self.les[x.name].fit(x))
        self.columns = cat_X_df.columns.tolist()
        return self

    def transform(self,**kwargs):
        cat_X_df = pd.DataFrame(kwargs['y'])
        res = cat_X_df.apply(lambda x: self.les[x.name].transform(x)).values
        return res

    def fit_transform(self,**kwargs):
        cat_X_df = pd.DataFrame(kwargs['y'])
        self.fit(**kwargs)
        return self.transform(**kwargs)

    def inverse_transform(self,**kwargs):
        transformed_y = kwargs['y']
        if not(isinstance(transformed_y, pd.DataFrame)):
            transformed_y = pd.DataFrame(kwargs['y'],columns=self.columns)
        return transformed_y.apply(lambda x: self.les[x.name].inverse_transform(x))


class MultiColumnsLabelBinarizer(LabelBinarizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lbs = defaultdict(lambda : LabelBinarizer(**kwargs))

    def fit(self,**kwargs):
        cat_X_df = pd.DataFrame(kwargs['y'])
        cat_X_df.apply(lambda x: self.lbs[x.name].fit(x))
        return self

    def transform(self,**kwargs):
        _X = pd.DataFrame(kwargs['y'])
        cols = _X.columns.tolist()
        transformed = [(cols[i] , self.lbs[cols[i]].transform(_X[cols[i]]) ) for i in range(len(cols))]
        res = np.concatenate([t[1] for t in transformed],axis=1)
        return res

    def fit_transform(self,**kwargs):
        cat_X_df = pd.DataFrame(kwargs['y'])
        self.fit(**kwargs)
        return self.transform(**kwargs)

    '''
    def inverse_transform(self,**kwargs):
        transformed_y = pd.DataFrame(kwargs['y'])
        return transformed_y.apply(lambda x: self.lbs[x.name].inverse_transform(x))
    '''


class LabelEncoBinarizer(LabelEncoder,LabelBinarizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.le = LabelEncoder(**kwargs)
        self.lb = LabelBinarizer(**kwargs)

    def fit(self,**kwargs):
        y = pd.Series(kwargs['y'])
        _X = self.le.fit_transform(y)
        self.lb.fit(_X)

        ####figure out mapping
        unqiue_ys = pd.Series(y.unique())
        self.mapping_table = pd.DataFrame({'src':unqiue_ys ,\
                                            'le':self.le.transform(unqiue_ys)})\
                                            .set_index('src')        
        lab2col = pd.DataFrame(self.lb.transform(self.mapping_table['le']),index=self.mapping_table.index.tolist())
        if len(unqiue_ys) > 2:
            self.is_binary = False
            self.mapping_table['lb'] = lab2col.apply(lambda row:row.idxmax() ,axis=1)
        elif len(unqiue_ys) == 2:
            self.is_binary = True
            self.mapping_table['lb'] = lab2col[0]
        self.mapping_table = self.mapping_table.sort_values(['le','lb'])
        return self

    def transform(self,**kwargs):
        y = pd.Series(kwargs['y'])
        transformed = self.lb.transform(self.le.transform(y))
        n_cols = transformed.shape[1]
        lb_mapping = self.mapping_table['lb']
        if self.is_binary:
            pos_class = lb_mapping[lb_mapping==1].index[0]
            res = pd.DataFrame(transformed,columns=['_IS_{}'.format(pos_class)])
        elif not(self.is_binary):
            df_columns = ['={}'.format(lb_mapping[lb_mapping==i].index[0]) for i in range(n_cols) ]
            res = pd.DataFrame(transformed,columns=df_columns)
        return res

    def fit_transform(self,**kwargs):
        self.fit(**kwargs)
        return self.transform(**kwargs)

    def inverse_transform(self,**kwargs):
        transformed_y = pd.DataFrame(kwargs['y'])
        return self.le.inverse_transform(self.lb.inverse_transform(transformed_y.values))

class MultiColumnsLabelEncoBinarizer(LabelEncoBinarizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lebs = defaultdict(lambda : LabelEncoBinarizer(**kwargs))

    def fit(self,**kwargs):
        cat_X_df = pd.DataFrame(kwargs['y'])
        if len(cat_X_df) > 0 and len(cat_X_df.columns) > 0: 
            cat_X_df.apply(lambda x: self.lebs[x.name].fit(y=x))
            self.transform(**kwargs)
        return self

    def transform(self,**kwargs):
        _X = pd.DataFrame(kwargs['y'])
        if len(_X) > 0 and len(_X.columns) > 0:
            cols = _X.columns.tolist()
            transformed = []

            for i in range(len(cols)):
                col_name = cols[i]
                t_df = self.lebs[col_name].transform(y=_X[col_name])

                t_df = t_df.rename(columns={c:'{}{}'.format(col_name,c) for c in t_df.columns.tolist()})
                transformed.append(t_df)

            res = pd.concat(transformed,axis=1)
            self.transformed_columns = res.columns.tolist()
            self.src_columns = cols
            return res
        return None

    def fit_transform(self,**kwargs):
        cat_X_df = pd.DataFrame(kwargs['y'])
        if len(cat_X_df) > 0 and len(cat_X_df.columns) > 0:
            self.fit(**kwargs)
            return self.transform(**kwargs)
        return None

    def inverse_transform(self,**kwargs):
        transformed_y = kwargs['y']
        if not(isinstance(transformed_y, pd.DataFrame)):
            transformed_y = pd.DataFrame(transformed_y,columns=self.transformed_columns)
        t_y_cols = transformed_y.columns.tolist()
        partitioned_dfs = pd.DataFrame({sc:self.lebs[sc].\
                                        inverse_transform(y=transformed_y[[c for c in t_y_cols if c.startswith(sc)]])\
                                        for sc in self.src_columns})

        return partitioned_dfs



if __name__ == '__main__':
    mle = MultiColumnsLabelEncoder()
    mlb = MultiColumnsLabelBinarizer()
    leb = LabelEncoBinarizer() 
    df = pd.DataFrame({'class':['A','B','C'],'is_smoker':['no','yes','no']})
    ''' df looks like this:
      class is_smoker
    0     A        no
    1     B       yes
    2     C        no    
    '''    

    A = mle.fit_transform(y=df)
    '''A looks like this:
    array([[0, 0],
           [1, 1],
           [2, 0]])    
    '''
    A_inv = mle.inverse_transform(y=A)
    '''A_inv looks like this:
      class is_smoker
    0     A        no
    1     B       yes
    2     C        no
    '''    

    B = mlb.fit_transform(y=A)
    '''B looks like this:
    array([[1, 0, 0, 0],
           [0, 1, 0, 1],
           [0, 0, 1, 0]])
    '''    
    # B_inv = mlb.fit_transform(y=B)
    

    ##########################binary labels
    src = ['B','B','A','B']
    C = leb.fit_transform(y=src)
    ''' C looks like this:
       _IS_B
    0      1
    1      1
    2      0
    3      1    
    '''    
    C_inv = leb.inverse_transform(y=C)
    '''C_inv looks like this
    array(['B', 'B', 'A', 'B'], dtype=object)
    '''

    ##########################multiclass labels
    leb = LabelEncoBinarizer()
    D = leb.fit_transform(y=src+['C'])
    ''' D looks like this:
       =A  =B  =C
    0   0   1   0
    1   0   1   0
    2   1   0   0
    3   0   1   0
    4   0   0   1
    '''
    D_inv = leb.inverse_transform(y=D)
    ''' D looks like this:
    ['B' 'B' 'A' 'B' 'C']
    '''
    mcleb = MultiColumnsLabelEncoBinarizer()
    E = mcleb.fit_transform(y=df)
    '''E looks like this:
       class=A  class=B  class=C  is_smoker_IS_yes
    0        1        0        0                 0
    1        0        1        0                 1
    2        0        0        1                 0    
    '''
    E_inv = mcleb.inverse_transform(y=E)

    ''' E_inv looks like this:
      class is_smoker
    0     A        no
    1     B       yes
    2     C        no
    '''