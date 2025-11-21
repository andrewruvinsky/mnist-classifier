make:
	clang++ -std=c++17 src/main.cpp -o build/model

clean:
	rm build/model
