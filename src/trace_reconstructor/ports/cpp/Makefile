CC = g++
CFLAGS = -std=c++17 -O3

OBJS = call_graph.o trace.o span.o
INCLUDE = -I. # place holder for now !TODO

SCHEMES = scheme fcfs

all: main.cpp main.h $(SCHEMES)
	$(CC) $(INCLUDE) $(CFLAGS) $(OBJS) -o $@ $< -ljsoncpp

$(SCHEMES): %: %.cpp $(OBJS) call_graph.cpp call_graph.h trace.cpp trace.h span.cpp span.h
	$(CC) $(INCLUDE) $(CFLAGS) -c $<

call_graph.o: call_graph.cpp call_graph.h trace.cpp trace.h span.cpp span.h
	$(CC) $(INCLUDE) $(CFLAGS) -c call_graph.cpp trace.cpp span.cpp

trace.o: trace.cpp trace.h span.cpp span.h
	$(CC) $(INCLUDE) $(CFLAGS) -c trace.cpp span.cpp

span.o: span.cpp span.h
	$(CC) $(INCLUDE) $(CFLAGS) -c span.cpp

clean:
	rm -f *.o $(SCHEMES) all
