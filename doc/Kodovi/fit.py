train_pool = Pool(train_X, train_y, cat_features = categoricalia)
test_pool = Pool(test_X, test_y, cat_features = categoricalia)

model.fit(
	train_pool,
	eval_set = test_pool,
	verbose = False,
	plot = True,
	early_stopping_rounds = 50
)

model.save('model')
