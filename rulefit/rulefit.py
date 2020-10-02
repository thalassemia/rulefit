"""Linear model of tree-based decision rules
This method implement the RuleFit algorithm
The module structure is the following:
- ``RuleCondition`` implements a binary feature transformation
- ``Rule`` implements a Rule composed of ``RuleConditions``
- ``RuleEnsemble`` implements an ensemble of ``Rules``
- ``RuleFit`` implements the RuleFit algorithm
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.linear_model import SGDClassifier, SGDRegressor
from functools import reduce
import time
import lightgbm as lgb

class RuleCondition():
    """Class for binary rule condition
    Warning: this class should not be used directly.
    """

    def __init__(self,
                 feature_index,
                 threshold,
                 operator,
                 support,
                 feature_name = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.operator = operator
        self.support = support
        self.feature_name = feature_name


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.feature_name:
            feature = self.feature_name
        else:
            feature = self.feature_index
        return "%s %s %s" % (feature, self.operator, self.threshold)

    def transform(self, X):
        """Transform dataset.
        Parameters
        ----------
        X: array-like matrix, shape=(n_samples, n_features)
        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, 1)
        """
        if self.operator == "<=":
            res =  1 * (X[:,self.feature_index] <= self.threshold)
        elif self.operator == ">":
            res = 1 * (X[:,self.feature_index] > self.threshold)
        return res

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash((self.feature_index, self.threshold, self.operator, self.feature_name))


class Winsorizer():
    """Performs Winsorization 1->1*
    Warning: this class should not be used directly.
    """
    def __init__(self,trim_quantile=0.0):
        self.trim_quantile=trim_quantile
        self.winsor_lims=None

    def train(self,X):
        # get winsor limits
        self.winsor_lims=np.ones([2,X.shape[1]])*np.inf
        self.winsor_lims[0,:]=-np.inf
        if self.trim_quantile>0:
            for i_col in np.arange(X.shape[1]):
                lower=np.percentile(X[:,i_col],self.trim_quantile*100)
                upper=np.percentile(X[:,i_col],100-self.trim_quantile*100)
                self.winsor_lims[:,i_col]=[lower,upper]

    def trim(self,X):
        X_=X.copy()
        X_=np.where(X>self.winsor_lims[1,:],np.tile(self.winsor_lims[1,:],[X.shape[0],1]),np.where(X<self.winsor_lims[0,:],np.tile(self.winsor_lims[0,:],[X.shape[0],1]),X))
        return X_

class FriedScale():
    """Performs scaling of linear variables according to Friedman et al. 2005 Sec 5
    Each variable is first Winsorized l->l*, then standardised as 0.4 x l* / std(l*)
    Warning: this class should not be used directly.
    """
    def __init__(self, winsorizer = None):
        self.scale_multipliers=None
        self.winsorizer = winsorizer

    def train(self,X):
        # get multipliers
        if self.winsorizer != None:
            X_trimmed= self.winsorizer.trim(X)
        else:
            X_trimmed = X

        scale_multipliers=np.ones(X.shape[1])
        for i_col in np.arange(X.shape[1]):
            num_uniq_vals=len(np.unique(X[:,i_col]))
            if num_uniq_vals>2: # don't scale binary variables which are effectively already rules
                scale_multipliers[i_col]=0.4/(1.0e-12 + np.std(X_trimmed[:,i_col]))
        self.scale_multipliers=scale_multipliers

    def scale(self,X):
        if self.winsorizer != None:
            return self.winsorizer.trim(X)*self.scale_multipliers
        else:
            return X*self.scale_multipliers


class Rule():
    """Class for binary Rules from list of conditions
    Warning: this class should not be used directly.
    """
    def __init__(self,
                 rule_conditions,prediction_value):
        self.conditions = set(rule_conditions)
        self.support = min([x.support for x in rule_conditions])
        self.prediction_value=prediction_value
        self.rule_direction=None
    def transform(self, X):
        """Transform dataset.
        Parameters
        ----------
        X: array-like matrix
        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, 1)
        """
        rule_applies = [condition.transform(X) for condition in self.conditions]
        return reduce(lambda x,y: x * y, rule_applies)

    def __str__(self):
        return  " & ".join([x.__str__() for x in self.conditions])

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return sum([condition.__hash__() for condition in self.conditions])

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


def extract_rules_from_tree(tree, feature_names=None):
    """Helper to turn a tree into as set of rules
    """
    rules = set()
    total_count = float(tree["internal_count"])
    def traverse_nodes(curr_tree=tree, split_index=0,
                       decision_type=None,
                       threshold=None,
                       feature=None,
                       support = None,
                       conditions=[]):
        if split_index != 0:
            if feature_names is not None:
                feature_name = feature_names[feature]
            else:
                feature_name = feature
            rule_condition = RuleCondition(feature_index=feature,
                                           threshold=threshold,
                                           operator=decision_type,
                                           support = support,
                                           feature_name=feature_name)
            new_conditions = conditions + [rule_condition]
        else:
            new_conditions = []
        ## if not terminal node
        if "leaf_index" not in curr_tree:
            feature = curr_tree["split_feature"]
            threshold = curr_tree["threshold"]
            support = curr_tree["internal_count"] / float(total_count)
            
            left_tree = curr_tree["left_child"]
            traverse_nodes(left_tree, curr_tree["split_index"], "<=", threshold, feature, support, new_conditions)

            right_tree = curr_tree["right_child"]
            traverse_nodes(right_tree, curr_tree["split_index"], ">", threshold, feature, support, new_conditions)
        else: # a leaf node
            if len(new_conditions) > 0:
                new_rule = Rule(new_conditions, curr_tree["leaf_value"])
                rules.update([new_rule])
            else:
                pass # tree only has a root node!
            return None
    
    traverse_nodes()

    return rules        


class RuleEnsemble():
    """Ensemble of binary decision rules
    This class implements an ensemble of decision rules that extracts rules from
    an ensemble of decision trees.
    Parameters
    ----------
    tree_list: List or array of DecisionTreeClassifier or DecisionTreeRegressor
        Trees from which the rules are created
    feature_names: List of strings, optional (default=None)
        Names of the features
    Attributes
    ----------
    rules: List of Rule
        The ensemble of rules extracted from the trees
    """
    def __init__(self,
                 tree_list,
                 feature_names=None):
        self.tree_list = tree_list
        self.feature_names = feature_names
        self.rules = set()
        self._extract_rules()
        self.rules=list(self.rules)

    def _extract_rules(self):
        """Recursively extract rules from each tree in the ensemble
        """
        for tree in self.tree_list:
            rules = extract_rules_from_tree(tree['tree_structure'], feature_names = self.feature_names)
            self.rules.update(rules)

    def transform(self, X,coefs=None):
        """Transform dataset.
        Parameters
        ----------
        X:      array-like matrix, shape=(n_samples, n_features)
        coefs:  (optional) if supplied, this makes the prediction
                slightly more efficient by setting rules with zero
                coefficients to zero without calling Rule.transform().
        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, n_out)
            Transformed dataset. Each column represents one rule.
        """
        rule_list=list(self.rules)
        if coefs is None :
            return np.array([rule.transform(X) for rule in rule_list]).T
        else: # else use the coefs to filter the rules we bother to interpret
            res= np.array([rule_list[i_rule].transform(X) for i_rule in np.arange(len(rule_list)) if coefs[i_rule]!=0]).T
            res_=np.zeros([X.shape[0],len(rule_list)])
            res_[:,coefs!=0]=res
            return res_
    def __str__(self):
        return (map(lambda x: x.__str__(), self.rules)).__str__()




class RuleFit(BaseEstimator, TransformerMixin):
    """Rulefit class
    Parameters
    ----------
        tree_size:      Number of terminal nodes in generated trees. If exp_rand_tree_size=True,
                        this will be the mean number of terminal nodes.
        sample_fract:   fraction of randomly chosen training observations used to produce each tree.
                        FP 2004 (Sec. 2)
        feature_fract:  fraction of randomly chosen features used to produce splits at each node
        num_trees:      total number of gradient boosted trees in final ensemble
        memory_par:     scale multiplier (shrinkage factor) applied to each new tree when
                        sequentially induced. FP 2004 (Sec. 2)
        rfmode:         'regress' for regression or 'classify' for binary classification.
        lin_standardise: If True, the linear terms will be standardised as per Friedman Sec 3.2
                        by multiplying the winsorised variable by 0.4/stdev.
        lin_trim_quantile: If lin_standardise is True, this quantile will be used to trim linear
                        terms before standardisation.
        model_type:     'r': rules only; 'l': linear terms only; 'rl': both rules and linear terms
        random_state:   Integer to initialise random objects and provide repeatability.
        tol:            The tolerance for the optimization for SGD Classifier/Regressor
        max_iter:       The maximum number of iterations for SGD Classifier/Regressor
        n_jobs:         Number of CPUs to use in ensemble generation and SGD linear fitting
        alpha:          Constant that multiplies the regularization term. The higher the value, 
                        the stronger the regularization. 
        lambda_l1:      L1 regularization parameter for LightGBM
        lambda_l2:      L2 regulatization parameter for LightGBM
        data_in_leaf:   minimum number of samples in each leaf
    Attributes
    ----------
    rule_ensemble: RuleEnsemble
        The rule ensemble
    feature_names: list of strings, optional (default=None)
        The names of the features (columns)
    """
    def __init__(
            self,
            tree_size=50,
            sample_fract=1.0,
            num_trees=100,
            memory_par=0.1,
            rfmode='regress',
            lin_trim_quantile=0.025,
            lin_standardise=True,
            model_type='rl',
            tol=0.0001,
            n_jobs=None,
            random_state=None,
            feature_fract=1.0,
            extra_trees=False,
            lambda_l1=0,
            lambda_l2=0,
            data_in_leaf=90,
            alpha=0.0001):
        self.rfmode=rfmode
        self.lin_trim_quantile=lin_trim_quantile
        self.lin_standardise=lin_standardise
        self.winsorizer=Winsorizer(trim_quantile=lin_trim_quantile)
        self.friedscale=FriedScale(self.winsorizer)
        self.stddev=None
        self.mean=None
        self.memory_par=memory_par
        self.num_trees=num_trees
        self.sample_fract=sample_fract
        self.tree_size = tree_size
        self.random_state=random_state
        self.model_type=model_type
        self.tol=tol
        self.n_jobs=n_jobs
        self.feature_fract=feature_fract
        self.lambda_l1=lambda_l1
        self.lambda_l2=lambda_l2
        self.data_in_leaf=data_in_leaf
        self.alpha=alpha

    def fit(self, X, y, X_test, y_test, feature_names=None):
        """Fit and estimate linear combination of rule ensemble

        """
        ## Make training set asymmetric so LightGBM can run
        X = X[0:X.shape[0] - 1, ]
        y = y[0:len(y) - 1]
        ## Enumerate features if feature names not provided
        N=X.shape[0]
        if feature_names is None:
            self.feature_names = ['feature_' + str(x) for x in range(0, X.shape[1])]
        else:
            self.feature_names=feature_names
        if 'r' in self.model_type:
            ## initialise tree generator
            if self.rfmode=='regress':
                param = {'num_leaves': self.tree_size, 'objective': 'regression', 'learning_rate': self.memory_par, 
                        'bagging_fraction': self.sample_fract,  'verbose': -1, 'bagging_freq': 1, 
                        'feature_fraction': self.feature_fract, 'min_data_in_leaf': self.data_in_leaf,
                        'force_row_wise': True}
                param['metric'] = ['l2', 'l1']
            else:
                param = {'num_leaves': self.tree_size, 'objective': 'binary', 'learning_rate': self.memory_par, 
                        'bagging_fraction': self.sample_fract,  'verbose': -1, 'bagging_freq': 1, 
                        'feature_fraction': self.feature_fract, 'min_data_in_leaf': self.data_in_leaf,
                        'force_row_wise': True}
                param['metric'] = ['auc', 'binary_logloss']
            # Prep training and validation datasets
            train_data = lgb.Dataset(X, label = y)
            validation_data = train_data.create_valid(X_test, label = y_test)
            ## fit tree generator
            self.trained_trees = lgb.train(param, train_data, self.num_trees, valid_sets=validation_data, 
                                            valid_names=['test'], feature_name=feature_names, verbose_eval=True)
            tree_list = self.trained_trees.dump_model()['tree_info']
            ## extract rules
            self.rule_ensemble = RuleEnsemble(tree_list = tree_list,
                                              feature_names=self.feature_names)
            ## concatenate original features and rules
            X_rules = self.rule_ensemble.transform(X)
        ## standardise linear variables if requested (for regression model only)
        if 'l' in self.model_type:
            ## standard deviation and mean of winsorized features
            self.winsorizer.train(X)
            winsorized_X = self.winsorizer.trim(X)
            self.stddev = np.std(winsorized_X, axis = 0)
            self.mean = np.mean(winsorized_X, axis = 0)
            if self.lin_standardise:
                self.friedscale.train(X)
                X_regn=self.friedscale.scale(X)
            else:
                X_regn=X.copy()

        ## Compile Training data
        X_concat=np.zeros([X.shape[0],0])
        if 'l' in self.model_type:
            X_concat = np.concatenate((X_concat,X_regn), axis=1)
        if 'r' in self.model_type:
            if X_rules.shape[0] >0:
                X_concat = np.concatenate((X_concat, X_rules), axis=1)

        ## fit linear model using SGD (stochastic gradient descent)
        y = np.float64(y)
        if self.rfmode=='regress':
            self.lscv=SGDRegressor(
                loss='squared_loss', penalty='l1', tol = self.tol, n_jobs=self.n_jobs, 
                random_state=self.random_state, max_iter=1000, early_stopping=True,
                alpha = self.alpha, n_iter_no_change=10, validation_fraction=0.1)
            self.lscv.fit(X_concat, y)
            self.coef_=self.lscv.coef_[0]
            self.intercept_=self.lscv.intercept_
        else:
            self.lscv=SGDClassifier(
                loss='log', penalty='l1', tol = self.tol, n_jobs=self.n_jobs, 
                random_state=self.random_state, max_iter=1000, early_stopping=True,
                alpha = self.alpha, n_iter_no_change=10, validation_fraction=0.1)
            start = time.time()
            self.lscv.fit(X_concat, y)
            self.coef_=self.lscv.coef_[0]
            self.intercept_=self.lscv.intercept_
        print(f"Finished fitting CV linear model in {time.time() - start}s")

        return self

    def predict(self, X):
        """Predict outcome for X
        """
        X_concat=np.zeros([X.shape[0],0])
        if 'l' in self.model_type:
            if self.lin_standardise:
                X_concat = np.concatenate((X_concat,self.friedscale.scale(X)), axis=1)
            else:
                X_concat = np.concatenate((X_concat,X), axis=1)
        if 'r' in self.model_type:
            rule_coefs=self.coef_[-len(self.rule_ensemble.rules):]
            if len(rule_coefs)>0:
                X_rules = self.rule_ensemble.transform(X,coefs=rule_coefs)
                if X_rules.shape[0] >0:
                    X_concat = np.concatenate((X_concat, X_rules), axis=1)
        return self.lscv.predict(X_concat)

    def predict_proba(self, X):
        """Predict outcome probability for X, if model type supports probability prediction method
        """
        X_concat=np.zeros([X.shape[0],0])
        if 'l' in self.model_type:
            if self.lin_standardise:
                X_concat = np.concatenate((X_concat,self.friedscale.scale(X)), axis=1)
            else:
                X_concat = np.concatenate((X_concat,X), axis=1)
        if 'r' in self.model_type:
            rule_coefs=self.coef_[-len(self.rule_ensemble.rules):]
            if len(rule_coefs)>0:
                X_rules = self.rule_ensemble.transform(X,coefs=rule_coefs)
                if X_rules.shape[0] >0:
                    X_concat = np.concatenate((X_concat, X_rules), axis=1)
        return self.lscv.predict_proba(X_concat)

    def transform(self, X=None, y=None):
        """Transform dataset.
        Parameters
        ----------
        X : array-like matrix, shape=(n_samples, n_features)
            Input data to be transformed. Use ``dtype=np.float32`` for maximum
            efficiency.
        Returns
        -------
        X_transformed: matrix, shape=(n_samples, n_out)
            Transformed data set
        """
        return self.rule_ensemble.transform(X)

    def get_rules(self, exclude_zero_coef=False, subregion=None):
        """Return the estimated rules
        Parameters
        ----------
        exclude_zero_coef: If True (default), returns only the rules with an estimated
                           coefficient not equalt to  zero.
        subregion: If None (default) returns global importances (FP 2004 eq. 28/29), else returns importance over
                           subregion of inputs (FP 2004 eq. 30/31/32).
        Returns
        -------
        rules: pandas.DataFrame with the rules. Column 'rule' describes the rule, 'coef' holds
               the coefficients and 'support' the support of the rule in the training
               data set (X)
        """

        n_features= len(self.coef_) - len(self.rule_ensemble.rules)
        rule_ensemble = list(self.rule_ensemble.rules)
        output_rules = []
        ## Add coefficients for linear effects
        for i in range(0, n_features):
            if self.lin_standardise:
                coef=self.coef_[i]*self.friedscale.scale_multipliers[i]
            else:
                coef=self.coef_[i]
            if subregion is None:
                importance = abs(coef)*self.stddev[i]
            else:
                subregion = np.array(subregion)
                importance = sum(abs(coef)* abs([ x[i] for x in self.winsorizer.trim(subregion) ] - self.mean[i]))/len(subregion)
            output_rules += [(self.feature_names[i], 'linear',coef, 1, importance)]

        ## Add rules
        for i in range(0, len(self.rule_ensemble.rules)):
            rule = rule_ensemble[i]
            coef=self.coef_[i + n_features]

            if subregion is None:
                importance = abs(coef)*(rule.support * (1-rule.support))**(1/2)
            else:
                rkx = rule.transform(subregion)
                importance = sum(abs(coef) * abs(rkx - rule.support))/len(subregion)

            output_rules += [(rule.__str__(), 'rule', coef,  rule.support, importance)]
        rules = pd.DataFrame(output_rules, columns=["rule", "type","coef", "support", "importance"])
        if exclude_zero_coef:
            rules = rules[rules.coef != 0]
        return rules
