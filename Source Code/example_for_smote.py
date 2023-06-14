from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
scaler = StandardScaler()
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RepeatedKFold
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from imblearn.over_sampling import SMOTE


#model = LogisticRegression(solver='liblinear', C=5, random_state=11,max_iter=1000)
#model = XGBClassifier(random_state=123, objective='binary:logistic')
model = RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    stratify=y,
                                                    random_state=11)

pipeline = imbpipeline(steps = [['smote', SMOTE(random_state=11)],
                                ['scaler', StandardScaler()],
                                ['classifier',model]])
stratified_kfold = StratifiedKFold(n_splits=10,
                                       shuffle=True,
                                       random_state=11)
#Repeated_kfold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
param_grid = {'classifier__max_depth': range (2, 10, 1), 
              'classifier__n_estimators': range(20, 220, 40),
              'classifier__min_samples_split' : [2, 5, 10],
              'classifier__bootstrap': [True, False],
              'classifier__max_features': ['auto', 'sqrt'],
              'classifier__min_samples_leaf': [1, 2, 4]
              #'classifier__max_features' : ['auto', 'sqrt']
              #'classifier__learning_rate': [0.1, 0.01, 0.05]
              #'classifier__colsample_bytree': [0.6, 0.8, 1.0],
              }
grid_search = GridSearchCV(estimator=pipeline,
                           param_grid=param_grid,
                           scoring='roc_auc',
                           cv=stratified_kfold,
                           n_jobs=-1)

rf_fit = grid_search.fit(X_train, y_train)
cv_score = grid_search.best_score_
# best_score_ is the average of all cv folds for a single combination of the parameters you specify in the tuned_params
feature_importances = rf_fit.best_estimator_._final_estimator.feature_importances_
test_score = grid_search.score(X_test, y_test)
y_pred = grid_search.predict(X_test)
print(f'Cross-validation score: {cv_score}\nTest score: {test_score}')