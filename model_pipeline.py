from sklearn import ensemble
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import cPickle as pickle


class ModelPipeline():

    def __init__(self, X=None, y=None):
        self.X = X
        self.y = y

    def metrics_scores(self):
        '''define the metrics to measure the performance of the model
        inputs: cleaned/engineered df and ylabel
        outputs: list of baseline scores (precision, recall, f1)
        '''
        y_true = self.y
        self.X['yhat'] = 1
        y_pred = self.X['yhat']
        basemodel_precision = metrics.precision_score(y_true, y_pred)
        basemodel_recall = metrics.recall_score(y_true, y_pred)
        basemodel_f1 = metrics.f1_score(y_true, y_pred)
        return([('basemodel_precision', basemodel_precision),
               ('basemodel_recall', basemodel_recall),
               ('basemodel_f1', basemodel_f1)])

    def run_logistic(self):
        logistic = LogisticRegression(penalty='l1')
        logistic.fit(self.X, self.y)
        log_score = cross_val_score(logistic, self.X, self.y,
                                    cv=5, scoring='f1_macro')
        return(zip('logistic', log_score))

    def othermodels(self):
        '''Calculate scores for all modesl except Logisitic.
        input: cleaned/engineered df
        output: return f1 scores from models
        '''
        # set models to run in pipeline
        sgd = SGDClassifier(loss="log", alpha=0.0001,
                                         learning_rate='optimal')
        svc = LinearSVC(C=1)
        randomforest = ensemble.RandomForestClassifier(max_depth=1,
                                                       max_features='auto',
                                                       n_estimators=150)
        adaboost = ensemble.AdaBoostClassifier(learning_rate=0.75,
                                               n_estimators=50)
        # save down sequence of models to run for future reference
        model_seq = ['sgd', 'svc', 'randomforest', 'adaboost']
        # define a pipeline
        pipe = Pipeline([('sgd', sgd),
                        ('svc', svc),
                        ('randomforest', randomforest),
                        ('adaboost', adaboost)])
        # fit models with train set
        pipe.fit(self.X, self.y)
        f1scores = cross_val_score(pipe, self.X, self.y,
                                   cv=5, scoring='f1_weighted')
        f1results = zip(model_seq, f1scores)
        recallscores = cross_val_score(pipe, self.X, self.y,
                                       cv=5, scoring='recall_weighted')
        recallresults = zip(model_seq, recallscores)
        precisionscores = cross_val_score(pipe, self.X, self.y,
                                          cv=5, scoring='precision_weighted')
        precisionresults = zip(model_seq, precisionscores)
        return("f1:", f1results,
               "recall:", recallresults,
               "precision:", precisionresults)

    def predict(self, X_test, y_test):
        X_all_training = self.X.append(X_test, ignore_index=True)
        y_all_training = self.y.append(y_test, ignore_index=True)
        randomforest = ensemble.RandomForestClassifier(max_depth=1,
                                                       max_features='auto',
                                                       n_estimators=150)
        randomforest.fit(X_all_training, y_all_training)
        f1scores = cross_val_score(randomforest,
                                   X_all_training, y_all_training,
                                   cv=5, scoring='f1_weighted')
        recallscores = cross_val_score(randomforest,
                                       X_all_training, y_all_training,
                                       cv=5, scoring='f1_weighted')
        precisionscores = cross_val_score(randomforest, X_all_training,
                                          y_all_training, cv=5,
                                          scoring='f1_weighted')
        print("f1:", f1scores,
              "recall:", recallscores,
              "precision:", precisionscores)

        # pickle randomforest fitted model and return that
        with open('data/final_model.pkl', 'w') as f:
            pickle.dump(randomforest, f)
        return "pickled model saved in data/final_model.pkl"

    def parameter_tuning(self, pipeline, params):
        # set models to run in pipeline
        sgd = SGDClassifier(loss='log',
                                         learning_rate='optimal', penalty='l1')
        svc = LinearSVC()
        randomforest = ensemble.RandomForestClassifier()
        adaboost = ensemble.AdaBoostClassifier()
        pipeline = Pipeline([('sgd', sgd),
                            ('svc', svc),
                            ('randomforest', randomforest),
                            ('adaboost', adaboost)])
        params = dict(sgd__alpha=[0.0001, 0.001, 0.01],
                      svc__C=[1, 10],
                      randomforest__n_estimators=[150, 300, 450],
                      randomforest__max_depth=[1, 3, None],
                      randomforest__max_features=['auto', 'sqrt', 'log2'],
                      adaboost__n_estimators=[50, 100, 150],
                      adaboost__learning_rate=[0.5, 0.75, 1.0])
        gs = GridSearchCV(pipeline,
                            params,
                            n_jobs=-1,
                            verbose=True,
                            cv=3,
                            scoring='recall_weighted')
        gs.fit(self.X, self.y)
        best_model = gs.best_estimator_
        best_params = gs.best_params_
        best_recall_score = gs.best_score_
        return best_model, best_params, best_recall_score
