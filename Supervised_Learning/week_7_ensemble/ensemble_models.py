from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble._forest import ForestClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from IPython.display import display
from sklearn.ensemble._forest import ForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample, gen_batches, check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.exceptions import NotFittedError

import numpy as np
import pandas as pd

def random_feature_subsets(array, batch_size, random_state=1234):
    """ Generate K subsets of the features in X """
    random_state = check_random_state(random_state)
    features = list(range(array.shape[1]))
    random_state.shuffle(features)
    for batch in gen_batches(len(features), batch_size):
        yield features[batch]


def random_feature_subsets(array, batch_size, random_state=1234):
    """ Generate K subsets of the features in X """
    random_state = check_random_state(random_state)
    features = list(range(array.shape[1]))
    random_state.shuffle(features)
    for batch in gen_batches(len(features), batch_size):
        yield features[batch]


class RotationTreeClassifier(DecisionTreeClassifier):
    # https://github.com/joshloyal/RotationForest/blob/master/rotation_forest/rotation_forest.py
    def __init__(self,
                 n_features_per_subset=3,
                 rotation_algo='pca',
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=1.0,
                 random_state=None,
                 max_leaf_nodes=None,
                 class_weight=None,
                 presort=False):

        self.n_features_per_subset = n_features_per_subset
        self.rotation_algo = rotation_algo
        self.presort = presort
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state)

    def rotate(self, X):
        if not hasattr(self, 'rotation_matrix'):
            raise NotFittedError('The estimator has not been fitted')

        return safe_sparse_dot(X, self.rotation_matrix)

    def pca_algorithm(self):
        """ Deterimine PCA algorithm to use. """
        if self.rotation_algo == 'pca':
            return PCA()
        else:
            raise ValueError("`rotation_algo` must be either "
                             "'pca' or 'randomized'.")

    def _fit_rotation_matrix(self, X):
        random_state = check_random_state(self.random_state)
        n_samples, n_features = X.shape
        self.rotation_matrix = np.zeros((n_features, n_features),
                                        dtype=np.float32)
        for i, subset in enumerate(
                random_feature_subsets(X, self.n_features_per_subset,
                                       random_state=self.random_state)):
            # take a 75% bootstrap from the rows
            x_sample = resample(X, n_samples=int(n_samples * 0.75), random_state=10 * i)
            pca = self.pca_algorithm()
            pca.fit(x_sample[:, subset])
            self.rotation_matrix[np.ix_(subset, subset)] = pca.components_

    def fit(self, X, y, sample_weight=None, check_input=True):
        self._fit_rotation_matrix(X)
        super().fit(self.rotate(X), y,
                    sample_weight, check_input)

    def predict_proba(self, X, check_input=True):
        return super().predict_proba(self.rotate(X), check_input)

    def predict(self, X, check_input=True):
        return super().predict(self.rotate(X),
                               check_input)

    def apply(self, X, check_input=True):
        return super().apply(self.rotate(X),
                             check_input)

    def decision_path(self, X, check_input=True):
        return super().decision_path(self.rotate(X),
                                     check_input)


class RotationForestClassifier(ForestClassifier):
    def __init__(self,
                 n_estimators=10,
                 criterion="gini",
                 n_features_per_subset=3,
                 rotation_algo='pca',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=1.0,
                 max_leaf_nodes=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super().__init__(
            base_estimator=RotationTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("n_features_per_subset", "rotation_algo",
                              "criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.n_features_per_subset = n_features_per_subset
        self.rotation_algo = rotation_algo
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes


class StackedModel(object):
    def __init__(self,
                 stacked_class=None,
                 stacked_args=None,
                 estimators_args=None,
                 random_state=0):

        self.random_state = random_state
        self.estimators_args = estimators_args
        self.estimators = {}
        self.stacked_class = stacked_class
        self.stacked_args = stacked_args
        self.stacked_model = None
        self.classes_ = []
        self.optimized_threshold = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0

    def fit_stacked(self, X_train, y_train, X_val, y_val):
        X_train_sub_probas = self.predict_subs(X_train)
        X_val_sub_probas = self.predict_subs(X_val)

        X_for_grid_cv = pd.concat([X_train_sub_probas, X_val_sub_probas]).reset_index(drop=True)
        y_for_grid_cv = pd.concat([y_train, y_val]).reset_index(drop=True)

        cv_for_grid_cv = [[list(range(0, y_train.shape[0])), list(range(y_train.shape[0], y_for_grid_cv.shape[0]))]]

        self.stacked_model = GridSearchCV(self.stacked_class(), self.stacked_args, cv=cv_for_grid_cv)
        self.stacked_model.fit(X_for_grid_cv, y_for_grid_cv)

        probs = self.stacked_model.predict_proba(X_val_sub_probas)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_val, probs)
        self.optimized_threshold = thresholds[(2 * (precision * recall) / (precision + recall)).argmax()]
        self.precision = precision.index(self.optimized_threshold)
        self.recall = recall.index(self.optimized_threshold)
        self.f1 = (2 * (precision * recall) / (precision + recall))

    def fit(self, X_train, y_train, X_val, y_val):
        self.classes_ = list(y_train.unique())
        X_train_sub, X_train_stacked, y_train_sub, y_train_stacked = train_test_split(X_train, y_train, test_size=0.2,
                                                                                      random_state=self.random_state)
        X_for_grid_cv = pd.concat([X_train_sub, X_val]).reset_index(drop=True)
        y_for_grid_cv = pd.concat([y_train_sub, y_val]).reset_index(drop=True)
        cv_for_grid_cv = [
            [list(range(0, y_train_sub.shape[0])), list(range(y_train_sub.shape[0], y_for_grid_cv.shape[0]))]]
        display(X_for_grid_cv)

        print(f'fit sub models')
        for estimator_name, estimator_args in self.estimators_args.items():
            print(f'fit {estimator_name}')
            self.estimators[estimator_name] = GridSearchCV(estimator_args['class'](), estimator_args['args'],
                                                           cv=cv_for_grid_cv)
            self.estimators[estimator_name].fit(X_for_grid_cv, y_for_grid_cv)

        print(f'fit stacked')
        self.fit_stacked(X_train_stacked, y_train_stacked, X_val, y_val)

    def predict_subs(self, X):
        stacked_X_preds = []
        for estimator_name, estimator_ in self.estimators.items():
            prob = estimator_.predict_proba(X)[:, 1]
            prob = pd.DataFrame(prob, columns=[estimator_name], index=X.index)
            stacked_X_preds.append(prob)
        stacked_X_preds_df = pd.concat(stacked_X_preds, axis=1)
        return stacked_X_preds_df

    def predict_proba(self, X):
        stacked_X_preds_df = self.predict_subs(X)
        return self.stacked_model.predict_proba(stacked_X_preds_df)

    def predict(self, X, check_input=True):
        stacked_X_preds_df = self.predict_subs(X)
        return self.stacked_model.predict(stacked_X_preds_df)



from dateutil.relativedelta import relativedelta
from sklearn.base import BaseEstimator

from sklearn.preprocessing import StandardScaler


class FeatureExtractor(object):
    def __init__(self, session_size=3):
        self.session_size = session_size
        self.session_keys = ['user_session', 'user_id']
        self.features_aggs = {'user_session': ['count'],
                              'weekday': ['first'],
                              'hour': ['first'],
                              'price': ['mean', 'std'],
                              'category_code': ['nunique', 'count'],
                              'category_id': ['nunique', 'count'],
                              'product_id': ['nunique'],
                              'brand': ['nunique', 'count']
                              }
        self.features_types = {'category_unique_ratio': 'ratio',
                               'product_unique_ratio': 'ratio',
                               'brand_unique_ratio': 'ratio',
                               'price_mean': 'float',
                               'price_std': 'float',
                               'category_id': 'list',
                               'brand': 'list',
                               'product_id': 'list',
                               'weekday_first': 'categorical',
                               'hour_first': 'categorical'}

        self.scaler = StandardScaler()
        self.apply_list = ['category_id', 'brand', 'category_code', 'product_id']

    def prep_raw(self, df):
        pd.set_option('mode.use_inf_as_na', False)
        first_sessions = df.sort_values('datetime').groupby(self.session_keys).head(3)

        first_sessions_features = first_sessions.groupby(self.session_keys).agg(self.features_aggs)
        first_sessions_features.columns = ['_'.join(col).strip() for col in first_sessions_features.columns.values]

        for c in self.apply_list:
            first_sessions_features[c] = first_sessions.groupby(self.session_keys)[c].apply(list)

        first_sessions_features['label'] = first_sessions.groupby(self.session_keys)['event_type'].apply(
            lambda x: x.eq('cart').any() | x.eq('purchase').any()).astype(int)
        first_sessions_features = first_sessions_features[first_sessions_features['user_session_count'] == 3]
        X = first_sessions_features.drop('label', axis=1)
        y = first_sessions_features['label']
        return X, y

    def get_features_by_type(self, f_type):
        return [f for f, t in self.features_types.items() if t == f_type]

    def fit(self, X, y=None):
        features = self.transform_(X)
        self.scaler.fit(features[self.get_features_by_type('float')])

    def transform(self, X):
        features = self.transform_(X)
        float_f = self.get_features_by_type('float')
        features[float_f] = self.scaler.transform(features[float_f])
        features = features.drop(self.apply_list, axis=1)
        return features

    def transform_(self, X):
        features = X.copy()
        features['category_unique_ratio'] = features['category_code_nunique'] / features['category_code_count']
        features['product_unique_ratio'] = features['product_id_nunique'] / features['user_session_count']
        features['brand_unique_ratio'] = features['brand_nunique'] / features['brand_count']
        pd.set_option('mode.use_inf_as_na', True)
        features = features.fillna(0)
        pd.set_option('mode.use_inf_as_na', False)
        return features


def evaluate(events_df, models, days_train, days_val, days_test, days_slide):
    cv = []
    train_start = events_df['date'].min()
    while True:
        train_end = train_start + relativedelta(days=days_slide)
        validation_end = train_end + relativedelta(days=days_val)
        test_end = validation_end + relativedelta(days=days_test)
        if test_end < events_df['date'].max():
            cv.append(
                dict(train_start=train_start, train_end=train_end, validation_end=validation_end,
                     test_end=test_end))
            train_start = train_start + relativedelta(days=days_train)
        else:
            break

    results = []
    for cv_ in cv:
        batch = cv_['validation_end']
        print(f'batch test start at {cv_["validation_end"]}')
        train_df = events_df[(events_df['date'] > cv_['train_start']) & (events_df['date'] <= cv_['train_end'])]
        validation_df = events_df[(events_df['date'] > cv_['train_end']) & (events_df['date'] <= cv_['validation_end'])]
        test_df = events_df[(events_df['date'] > cv_['validation_end']) & (events_df['date'] <= cv_['test_end'])]

        print('prep data')
        feature_extractor = FeatureExtractor()
        train_X, train_y = feature_extractor.prep_raw(train_df)
        val_X, val_y = feature_extractor.prep_raw(validation_df)
        test_X, test_y = feature_extractor.prep_raw(test_df)

        feature_extractor = FeatureExtractor()
        print('extract features')
        feature_extractor.fit(train_X, train_y)
        train_features = feature_extractor.transform(train_X)
        val_features = feature_extractor.transform(val_X)
        test_features = feature_extractor.transform(test_X)

        X_for_grid_cv = pd.concat([train_features, val_features]).reset_index(drop=True)
        y_for_grid_cv = pd.concat([train_y, val_y]).reset_index(drop=True)
        cv_for_grid_cv = [[list(range(0, train_y.shape[0])), list(range(train_y.shape[0], y_for_grid_cv.shape[0]))]]
        for model_name, model_params in models.items():

            print(f'fit {model_name}')
            if isinstance(model_params['class'](), BaseEstimator):
                clf = GridSearchCV(model_params['class'](), model_params['args'], cv=cv_for_grid_cv)
                clf.fit(X_for_grid_cv, y_for_grid_cv)
                best_params = clf.best_params_
            else:
                clf = model_params['class'](**model_params['args'])
                clf.fit(train_features, train_y, val_features, val_y)
                best_params = ""

            probs_val = clf.predict_proba(val_features)[:, 1]
            precision, recall, thresholds = precision_recall_curve(val_y, probs_val)  # TODO change to validation
            optimized_threshold = thresholds[(2 * (precision * recall) / (precision + recall)).argmax()]

            preds = clf.predict(test_features)
            probs = clf.predict_proba(test_features)[:, 1]
            preds_optimized = [probs > optimized_threshold][0].astype(int)

            model_res = dict(batch=str(batch), model_name=model_name, best_params=best_params,
                             accuracy_score=accuracy_score(test_y, preds), f1_score=f1_score(test_y, preds),
                             precision=precision_score(test_y, preds), recall=recall_score(test_y, preds),
                             f1_score_optimized=f1_score(test_y, preds_optimized),
                             optimized_threshold=optimized_threshold

                             )
            results.append(model_res)
    return results

if __name__ == '__main__':
    events_df = pd.read_csv("/Users/liorsidi/Downloads/events.csv")
    events_df['datetime'] = events_df['event_time'].apply(lambda x: pd.to_datetime(x))
    events_df['date'] = events_df['datetime'].apply(lambda x: x.date())
    events_df['weekday'] = events_df['datetime'].apply(lambda x: x.weekday)
    events_df['hour'] = events_df['datetime'].apply(lambda x: x.hour)
    estimators_args_pack = {
        "LogisticRegression": {"class": LogisticRegression, "args": {'penalty': ['l2']}},
        "DecisionTreeClassifier": {"class": DecisionTreeClassifier,
                                   "args": {"criterion": ["entropy", "gini"], "min_samples_leaf": [5, 15, 20]}},
        "AdaBoostClassifier": {"class": AdaBoostClassifier,
                               "args": {"n_estimators": [20, 50, 100], "random_state": [0]},
                               "hyper": {"learning_rate": [0.1, 0.25, 0.5, 0.75, 1.0]}},
        "RandomForestClassifier": {"class": RandomForestClassifier,
                                   "args": {"criterion": ["entropy", "gini"], "n_estimators": [50, 100],
                                            "max_depth": [None, 10]}}}

    submodels_pack = {"StackedModel": {"class": StackedModel, "args": dict(stacked_class=LogisticRegression,
                                                                           stacked_args=estimators_args_pack[
                                                                               'LogisticRegression'],
                                                                           estimators_args=estimators_args_pack)}}

    stacked_results = evaluate(events_df, submodels_pack, 2, 2, 2, 28)