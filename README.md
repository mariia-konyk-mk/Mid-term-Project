# Mid-term-Project
Mid-term Project:Прогнозування відкриття банківського депозиту 
Проєкт присвячений побудові модель для передбачення, чи оформить клієнт в банку строковий депозит (term deposit)
Це задача бінарної класифікації, де цільова змінна y вказує на успіх (yes) або невдачу (no) контакту.

Що було зроблено:
1. Проведено Exploratory Data Analysis і висунуто гіпотези щодо впливу окремих ознак на цільову ознаку y.
2. Проведено препроцесинг даних для подальшої їх передачі в модель:
-обробка категоріальних змінних 
-заповнення пропущених значень
-виявлення outliers і прийнятто рішень, що з ними робити
-створенно додаткових ознак, які на ваш погляд поліпшать якість моделей ML.
3. Натренеровано 4 різні типи моделей машинного навчання, з яких
-Logistic Regression
-kNN
-Decision Tree
-XGBoost.
4.Для алгоритму XGBoost виконано процедуру тюнингу гіперпараметрів двома способами:
- Використано Hyperopt (Bayesian Optimization),  виявлено оптимальні гіперпараметри
- Sklearn: Randomized Search.
5. Виведено важливість ознак для моделі, які показала себе найкраще і описано
  їх.
6. Для найкращої моделі проведено аналіз впливу ознак на передбачення з допомогою бібліотеки SHAP.
7. Проведено аналіз записів, в яких модель помиляється і на основі нього зазначено, яким чином можна поліпшити наявне рішення аби уникати наявних помилок
8. Створено таблицю з порівнянням якості моделей 

| Модель | Гіперпараметри | Train AUC | Val AUC | Test AUC | Коментар |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression** |solver='liblinear',class_weight='balanced',random_state=42| 0.93 | 0.93 | 0.94 | Хороша модель. Можна використовувати як базову, відсутні ознаки перенавчання. |
| **kNN** | n_neighbors=np.int64(48) (GridSearchCV)| 0.86 | 0.89 | 0.81 | Хороша модель, але є ознаки перенавчання. |
| **Decision Tree** | criterion='entropy', max_depth=np.int64(16),
                       max_leaf_nodes=np.int64(14), min_samples_leaf=2,
                       min_samples_split=20, random_state=42 
(RandomizedSearch) | 0.93 | 0.93 | 0.94 | Хороша. На рівні з регресією. Дає чітку логіку розподілу. |
| **XGBoost (Hyperopt)** | {'colsample_bytree': np.float64(0.8259351675636171), 'gamma': np.float64(0.24487485092781786), 'learning_rate': np.float64(0.08863921968093902), 'max_depth': 7, 'min_child_weight': 4, 'n_estimators': 350, 'reg_alpha': np.float64(0.8980006701651116), 'reg_lambda': np.float64(0.1854974501749974), 'subsample': np.float64(0.8189650451935616)} (Hyperopt)| 0.99 | 0.94 | 0.95 | Дуже гарна модель з високою точністю. |
| **XGBoost (Random Search)** | base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=0.7, device='cuda', early_stopping_rounds=None,
              enable_categorical=True, eval_metric=None, feature_types=None,
              feature_weights=None, gamma=0.1, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=0.05, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None,
              max_depth=np.int64(6), max_leaves=None, min_child_weight=None,
              missing=nan, monotone_constraints=None, multi_strategy=None,
              n_estimators=200, n_jobs=None, num_parallel_tree=None, ...) (Randomized Search) | **0.97** | **0.95** | **0.96** | **Найкраща модель.** Максимальна точність. Рекомендовано до впровадження. |
