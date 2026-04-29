CXX := g++
CXXFLAGS := -std=c++17 -O3
TARGET := build/model
SOURCES := src/*.cpp

make:
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)
