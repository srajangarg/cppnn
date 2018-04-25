# dummy
all:

	@clang-format-3.8 -i *.h *.cpp
	@g++ test.cpp -std=c++11 -O3 -otest -Wextra -Wpedantic

clean:

	rm test
