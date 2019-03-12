import numpy as np
import pandas as pd
import pdb
from sklearn.tree import DecisionTreeClassifier
from copy import deepcopy

class DecisionTreeClassifierWithDecisionRules(DecisionTreeClassifier):
    def wrap_DecisionTreeClassifier(dt_clf):
        clf_clone = deepcopy(dt_clf)
        clf2 = clf_clone
        clf2.__class__ = DecisionTreeClassifierWithDecisionRules        
        return clf2


    def decision_rules_report(self,X):
        Y_PREFIX = '__y_'
        Y_COLUMN_TMPL = Y_PREFIX + '{}'
        LEAVE_ID_COLUMN = '__leave_id'

        DECISION_RULES_CONDITION_COL = 'condition'
        DECISION_RULES_LEAF_ID_COL = 'leaf_id'

        n_nodes = self.tree_.node_count
        children_left = self.tree_.children_left
        children_right = self.tree_.children_right
        feature = self.tree_.feature
        threshold = self.tree_.threshold

        X = pd.DataFrame(X)
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
        
        decision_rules = []
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
            
            tobeinserted = {DECISION_RULES_CONDITION_COL:' AND '.join(conditions),DECISION_RULES_LEAF_ID_COL:leave_id}
            for col in ys_cols:
                tobeinserted[col.replace(Y_PREFIX,'')] = current_row[col]
            
            decision_rules.append( tobeinserted )

        self.decision_rules = pd.DataFrame(decision_rules)
        return self.decision_rules
        


            
        

if __name__ == '__main__':
    X = pd.DataFrame(np.random.randn(1000, 50),columns=['feature_{}'.format(i) for i in range(50)])
    y = ['A'] * 300 + ['B'] * 300 + ['C'] * 400

    clf2 = DecisionTreeClassifierWithDecisionRules(random_state=0,criterion='entropy',**{'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 0.01})
    clf2.fit(X,y)
    print(clf2.decision_rules_report(X))


    #######converting existing DecisionTreeClassifier into DecisionTreeClassifierWithDecisionRules
    from copy import deepcopy
    clf = DecisionTreeClassifier(random_state=0,criterion='entropy',**{'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 0.01})
    clf.fit(X,y)

    clf2 = DecisionTreeClassifierWithDecisionRules.wrap_DecisionTreeClassifier(clf)

    print( clf2.decision_rules_report(X) )



    