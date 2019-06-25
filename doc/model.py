model = cb.CatBoostClassifier(
	iterations = 1000,
	learning_rate = 0.873,
	depth = 9,
	l2_leaf_reg = 743.5,
	border_count = 168,
	od_type = 'Iter',
	leaf_estimation_method = 'Newton',
	random_seed = 934,
	random_strength = 1.419,
	bagging_temperature = 0.415,
	task_type ='GPU',
	sampling_unit = 'Group'
)
