CXX = mpic++

CXXFLAGS = -O3 -march=native -flto -std=c++11 -funroll-loops

TARGET = check

SOURCES = check.cpp template.cpp

.PHONY: compile clean

compile:
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET)
