install:
	make -C src/ wynona

test: install
	pytest test/test.py

clean:
	make -C src/ clean

.PHONY: install clean