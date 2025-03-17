
lint:
	ruff format
	ruff check --fix

run:
	python3 diamondtime.py config.yaml --series-length 2 --constraints constraints.yaml