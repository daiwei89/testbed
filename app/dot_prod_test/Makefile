VEC_PROD_TEST := $(shell readlink $(dir $(lastword $(MAKEFILE_LIST))) -f)
PETUUM_ROOT = $(VEC_PROD_TEST)/../../

include $(PETUUM_ROOT)/defns.mk

DOT_PROD_TEST_SRC = $(wildcard $(VEC_PROD_TEST)/src/*.cpp)
DOT_PROD_TEST_HDR = $(wildcard $(VEC_PROD_TEST)/src/*.hpp)
DOT_PROD_TEST_BIN = $(VEC_PROD_TEST)/bin
DOT_PROD_TEST_OBJ = $(DOT_PROD_TEST_SRC:.cpp=.o)
NDEBUG = -DNDEBUG

all: $(DOT_PROD_TEST_BIN)/test_main

$(DOT_PROD_TEST_BIN):
	mkdir -p $(DOT_PROD_TEST_BIN)

$(DOT_PROD_TEST_BIN)/test_main: $(DOT_PROD_TEST_OBJ) $(PETUUM_PS_LIB) $(PETUUM_ML_LIB) $(DOT_PROD_TEST_BIN)
	$(PETUUM_CXX) $(PETUUM_CXXFLAGS) $(PETUUM_INCFLAGS) \
	$(DOT_PROD_TEST_OBJ) $(PETUUM_PS_LIB) $(PETUUM_ML_LIB) $(PETUUM_LDFLAGS) -o $@

$(DOT_PROD_TEST_OBJ): %.o: %.cpp $(DOT_PROD_TEST_HDR)
	$(PETUUM_CXX) $(NDEBUG) $(PETUUM_CXXFLAGS) -Wno-unused-result \
		$(PETUUM_INCFLAGS) -c $< -o $@

clean:
	rm -rf $(DOT_PROD_TEST_OBJ)
	rm -rf $(DOT_PROD_TEST_BIN)

.PHONY: clean
