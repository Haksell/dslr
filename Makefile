install:
	/usr/bin/pip install --upgrade matplotlib pandas PyQT5 pytest scikit-learn seaborn

clean:
	rm -rf __pycache__ .pytest_cache

test:
	@pytest -rA -vv -W ignore::DeprecationWarning -W ignore::RuntimeWarning