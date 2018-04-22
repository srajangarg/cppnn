# dummy
all:

	@clang-format-3.8 -i *.h *.cpp
	@g++ test.cpp -std=c++11 -otest

clean:

	rm test
