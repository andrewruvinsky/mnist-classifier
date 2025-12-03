make:
	g++ -std=c++17 src/*.cpp -o build/model

clean:
	rm build/model
