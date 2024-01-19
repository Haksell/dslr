clean:
	rm -rf __pycache__ .pytest_cache houses.csv parameters.json

install:
	/usr/bin/pip install --upgrade matplotlib pandas PyQT5 pytest scikit-learn seaborn

run:
	@./logreg_train.py datasets/dataset_top1200.csv
	@./logreg_predict.py datasets/dataset_bottom400.csv parameters.json

test:
	@pytest -rA -vv -W ignore::DeprecationWarning -W ignore::RuntimeWarning