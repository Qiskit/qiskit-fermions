.PHONY: cext testc pyext pyinstall testpython echo_pyexport testrust doctest cdoc docs docsclean lint style

export QISKIT_CEXT_INSTALL_METHOD := path
export QISKIT_CEXT_PATH := $(shell python -c "import os; import qiskit; print(os.path.dirname(qiskit._accelerate.__file__) + '/..')")
export BINDGEN_EXTRA_CLANG_ARGS := "-I$(shell python -c "import sysconfig; print(sysconfig.get_path('include'))")"

C_DIR_OUT = dist/c
C_DIR_LIB = $(C_DIR_OUT)/lib
C_DIR_INCLUDE = $(C_DIR_OUT)/include
C_DIR_TEST_BUILD = tests/c/build
# Whether this is target/debug or target/release depends on the flags in the
# `cheader` recipe.  For now, they're just hardcoded.
C_CARGO_TARGET_DIR = target/release
ifeq ($(OS), Windows_NT)
	C_LIB_CARGO_FILENAME=qiskit_fermions_cext.dll
else ifeq ($(shell uname), Darwin)
	C_LIB_CARGO_FILENAME=libqiskit_fermions_cext.dylib
else
	# ... probably.
	C_LIB_CARGO_FILENAME=libqiskit_fermions_cext.so
endif
C_LIB_CARGO_PATH=$(C_CARGO_TARGET_DIR)/$(C_LIB_CARGO_FILENAME)

C_LIBQISKIT_FERMIONS=$(C_DIR_LIB)/$(subst _cext,,$(C_LIB_CARGO_FILENAME))
C_QISKIT_FERMIONS_H=$(C_DIR_INCLUDE)/qiskit_fermions.h

$(C_DIR_LIB):
	mkdir -p $(C_DIR_LIB)

$(C_DIR_INCLUDE):
	mkdir -p $(C_DIR_INCLUDE)/

cext: $(C_DIR_LIB) $(C_DIR_INCLUDE)
	LD_LIBRARY_PATH="${QISKIT_CEXT_PATH}/dist/c/lib:${LD_LIBRARY_PATH}" cargo rustc --release --crate-type cdylib -p qiskit-fermions-cext
	cp $(C_LIB_CARGO_PATH) $(C_DIR_LIB)/$(subst _cext,,$(C_LIB_CARGO_FILENAME))
	cp target/qiskit_fermions.h $(C_DIR_INCLUDE)/qiskit_fermions.h

testc: cext
	LD_LIBRARY_PATH="${QISKIT_CEXT_PATH}/dist/c/lib:${LD_LIBRARY_PATH}" cmake -S. -B$(C_DIR_TEST_BUILD)
	LD_LIBRARY_PATH="${QISKIT_CEXT_PATH}/dist/c/lib:${LD_LIBRARY_PATH}" cmake --build $(C_DIR_TEST_BUILD)
	LD_LIBRARY_PATH="${QISKIT_CEXT_PATH}/dist/c/lib:${LD_LIBRARY_PATH}" ctest -V -C Debug --test-dir $(C_DIR_TEST_BUILD)

pyext:
	LD_LIBRARY_PATH="${QISKIT_CEXT_PATH}/qiskit:${LD_LIBRARY_PATH}" cargo run --bin stub_gen -p qiskit-fermions-pyext --no-default-features
	LD_LIBRARY_PATH="${QISKIT_CEXT_PATH}/qiskit:${LD_LIBRARY_PATH}" python setup.py build_rust --inplace --release

pyinstall: pyext
	LD_LIBRARY_PATH="${QISKIT_CEXT_PATH}/qiskit:${LD_LIBRARY_PATH}" pip install -e ".$(DEPS)"

testpython: pyext
	LD_LIBRARY_PATH="${QISKIT_CEXT_PATH}/qiskit:${LD_LIBRARY_PATH}" python setup.py build_rust --inplace --release
	LD_LIBRARY_PATH="${QISKIT_CEXT_PATH}/qiskit:${LD_LIBRARY_PATH}" python -m pytest -s --doctest-plus --doctest-glob "*.pyi"

echo_pyexport:
	@echo export QISKIT_CEXT_INSTALL_METHOD="${QISKIT_CEXT_INSTALL_METHOD}"
	@echo export QISKIT_CEXT_PATH="${QISKIT_CEXT_PATH}"
	@echo export LD_LIBRARY_PATH="${QISKIT_CEXT_PATH}/qiskit:${LD_LIBRARY_PATH}"
	@echo export BINDGEN_EXTRA_CLANG_ARGS="${BINDGEN_EXTRA_CLANG_ARGS}"

testrust:
	LD_LIBRARY_PATH="${QISKIT_CEXT_PATH}/dist/c/lib:${LD_LIBRARY_PATH}" cargo test -p qiskit-fermions-core --no-default-features

doctest: pyext
	LD_LIBRARY_PATH="${QISKIT_CEXT_PATH}/qiskit:${LD_LIBRARY_PATH}" python -m pytest docs/ -s --doctest-plus --doctest-only --doctest-glob "*.rst"

cdoc: cext
	doxygen docs/Doxyfile

docs: cdoc pyext
	LD_LIBRARY_PATH="${QISKIT_CEXT_PATH}/qiskit:${LD_LIBRARY_PATH}" sphinx-build -W -j auto -T -E --keep-going -b html docs/ docs/_build/html

docsclean:
	rm -rf docs/stubs/ docs/_build docs/xml

lint:
	LD_LIBRARY_PATH="${QISKIT_CEXT_PATH}/qiskit:${LD_LIBRARY_PATH}" cargo clippy
	tox -e lint

style:
	LD_LIBRARY_PATH="${QISKIT_CEXT_PATH}/qiskit:${LD_LIBRARY_PATH}" cargo fmt
	tox -e style
	clang-format --style="file:.clang-format" -i tests/c/*.c tests/c/*.h
