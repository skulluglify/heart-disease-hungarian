
```python

from sklearn.ensemble import RandomForestClassifier

n_estimators: int
criterion: ['gini', 'entropy', 'log_loss']
max_depth: int | None = None
min_samples_split: int
min_samples_leaf: int
min_weight_fraction_leaf: float
max_features: ['sqrt', 'log2', None]
max_leaf_nodes: int
min_impurity_decrease: float
bootstrap: bool
oob_score: bool
n_jobs: int | None = None
random_state: int
verbose: int = 0
warm_start: bool = False
class_weight: ['balanced', 'balanced_subsample']
ccp_alpha: float = 0.0 <= n
max_samples: int


from sklearn.linear_model import LinearRegression

fit_intercept: bool = True
copy_X: bool = True
n_jobs: int | None = None
positive: bool = False


from sklearn.neighbors import KNeighborsClassifier

n_neighbors: int = 5
weights: ['uniform', 'distance']
algorithm: ['ball_tree', 'kd_tree', 'brute']
leaf_size: int = 30
p: float = 2.0
metric: str = 'minkowski'
metric_params: dict | None = None
n_jobs: int | None = None


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

criterion: ['gini', 'entropy', 'log_loss']
splitter: ['best', 'random']
max_depth: int | None = None
min_samples_split: int = 2
min_samples_leaf: int = 1
min_weight_fraction_leaf: float = 0.0
max_features: int | str = ['sqrt', 'log2'] | None = None
random_state: int | None = None
max_leaf_nodes: int | None = None
min_impurity_decrease: float = 0.0
class_weight: dict | list | 'balanced' | None = None
ccp_alpha: float = 0.0 <= n

criterion: ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
splitter: ['best', 'random']
max_depth: int | None = None
min_samples_split: int = 2
min_samples_leaf: int = 1
min_weight_fraction_leaf: float = 0.0
max_features: int | str = ['sqrt', 'log2']
random_state: int | None = None
max_leaf_nodes: int
min_impurity_decrease: float = 0.0
ccp_alpha: float = 0.0 <= n


from xgboost import XGBClassifier

booster: ['gbtree', 'gblinear', 'dart']
device = ['cpu', 'cuda', 'gpu']
verbosity: int = 0
validate_parameters: bool = False
nthread: int | None = None
disable_default_eval_metric: int = 0
```
