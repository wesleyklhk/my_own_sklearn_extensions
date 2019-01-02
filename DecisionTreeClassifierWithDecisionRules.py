import numpy as np
import pandas as pd
import pdb
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeClassifierWithDecisionRules(DecisionTreeClassifier):
    def fit(self,**kwargs):
        super().fit(**kwargs)
        n_nodes = self.tree_.node_count
        children_left = self.tree_.children_left
        children_right = self.tree_.children_right
        feature = self.tree_.feature
        threshold = self.tree_.threshold

        X = pd.DataFrame(kwargs['X'])
        y = kwargs['y']
        cols = X.columns.tolist()
        tmp = X.copy()
        tmp['__leave_id'] = self.apply(X[cols])
        probs_df = pd.DataFrame( self.predict_proba(X[cols]) , columns=self.classes_ , index=X.index)
        probs_df = probs_df.rename(columns={col:'__y_{}'.format(col) for col in probs_df.columns.tolist()})
        tmp = pd.concat([tmp.reset_index(drop=True),probs_df.reset_index(drop=True)],axis=1)
        all_leaves = tmp['__leave_id'].unique().tolist()
        all_leaves.sort()
        
        all_paths_with_x = []
        for lid in all_leaves:
            all_paths_with_x.append( tmp[tmp['__leave_id'] == lid].head(1) )

        all_paths_with_x_and_probs = pd.concat(all_paths_with_x).reset_index(drop=True)

        node_indicator = self.decision_path(all_paths_with_x_and_probs[cols])
        
        case_stmt_src = []
        for sample_id in range(len(all_paths_with_x_and_probs)):
            current_row = all_paths_with_x_and_probs.iloc[sample_id]
            leave_id = current_row['__leave_id']
            ys_cols = [col for col in probs_df.columns.tolist() if col.startswith('__y_')]
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
                tobeinserted[col.replace('__y_','')] = current_row[col]
            
            case_stmt_src.append( tobeinserted )

        self.case_stmt_src = pd.DataFrame(case_stmt_src)
        return self
            
        


X = pd.DataFrame(np.random.randn(100, 10),columns=['feature_{}'.format(i) for i in range(10)])
y = ['A'] * 30 + ['B'] * 30 + ['C'] * 40

clf2 = DecisionTreeClassifierWithDecisionRules(random_state=0,criterion='entropy',**{'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 0.01})
clf2.fit(X=X,y=y)
print(clf2.case_stmt_src)