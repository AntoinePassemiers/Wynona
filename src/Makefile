wynona:
	python setup.py install

build:
	python setup.py build_ext --inplace

clean:
	rm -f wynona/*.c
	rm -f wynona/**/*.c
	rm -f wynona/*.so
	rm -f wynona/**/*.so
	rm -fr build/

.PHONY: wynona build clean