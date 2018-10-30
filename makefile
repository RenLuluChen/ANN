all:
	g++-7.2.0 -std=c++17 main.cpp neural_network.cpp neuron.cpp
debug:
	g++-7.2.0 -std=c++17 -g main.cpp neural_network.cpp neuron.cpp -o debug
run:
	./a.out
clean:
	rm a.out