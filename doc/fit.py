model.fit(
	train_pool,
	eval_set = test_pool,
	verbose = False,
	plot = True,
	early_stopping_rounds = 50
)
