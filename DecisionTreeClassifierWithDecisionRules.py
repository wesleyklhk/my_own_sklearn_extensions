import numpy as np
import pandas as pd
import pdb
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeClassifierWithDecisionRules(DecisionTreeClassifier):
    def fit(self,**kwargs):
        super().fit(**kwargs)
        Y_PREFIX = '__y_'
        Y_COLUMN_TMPL = Y_PREFIX + '{}'
        LEAVE_ID_COLUMN = '__leave_id'

        n_nodes = self.tree_.node_count
        children_left = self.tree_.children_left
        children_right = self.tree_.children_right
        feature = self.tree_.feature
        threshold = self.tree_.threshold

        X = pd.DataFrame(kwargs['X'])
        y = kwargs['y']
        cols = X.columns.tolist()
        tmp = X.copy()
        tmp[LEAVE_ID_COLUMN] = self.apply(X[cols])
        probs_df = pd.DataFrame( self.predict_proba(X[cols]) , columns=self.classes_ , index=X.index)
        probs_df = probs_df.rename(columns={col:Y_COLUMN_TMPL.format(col) for col in probs_df.columns.tolist()})
        tmp = pd.concat([tmp.reset_index(drop=True),probs_df.reset_index(drop=True)],axis=1)
        all_leaves = tmp[LEAVE_ID_COLUMN].unique().tolist()
        all_leaves.sort()
        
        all_paths_with_x = []
        for lid in all_leaves:
            all_paths_with_x.append( tmp[tmp[LEAVE_ID_COLUMN] == lid].head(1) )

        all_paths_with_x_and_probs = pd.concat(all_paths_with_x).reset_index(drop=True)

        node_indicator = self.decision_path(all_paths_with_x_and_probs[cols])
        
        case_stmt_src = []
        for sample_id in range(len(all_paths_with_x_and_probs)):
            current_row = all_paths_with_x_and_probs.iloc[sample_id]
            leave_id = current_row[LEAVE_ID_COLUMN]
            ys_cols = [col for col in probs_df.columns.tolist() if col.startswith(Y_PREFIX)]
            node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                                node_indicator.indptr[sample_id + 1]]
            conditions = []
            subcondition_tmpl = '[{}] {} {}'
            for node_id in node_index:
                if leave_id == node_id:
                    continue

                if ( current_row[cols[feature[node_id]]] <= threshold[node_id]):
                    threshold_sign = "<="
                else:
                    threshold_sign = ">"


                conditions.append( subcondition_tmpl.format(cols[feature[node_id]] , threshold_sign , threshold[node_id]) )
            
            tobeinserted = {'when_clause':' AND '.join(conditions),'leaf_id':leave_id}
            for col in ys_cols:
                tobeinserted[col.replace(Y_PREFIX,'')] = current_row[col]
            
            case_stmt_src.append( tobeinserted )

        self.case_stmt_src = pd.DataFrame(case_stmt_src)
        return self
            
        


X = pd.DataFrame(np.random.randn(1000, 50),columns=['feature_{}'.format(i) for i in range(50)])
y = ['A'] * 300 + ['B'] * 300 + ['C'] * 400

clf2 = DecisionTreeClassifierWithDecisionRules(random_state=0,criterion='entropy',**{'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 0.01})
clf2.fit(X=X,y=y)
print(clf2.case_stmt_src)