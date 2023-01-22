CC = g++
CFLAGS = -std=c++17 -O3

OBJS = trace.o span.o call_graph.o
INCLUDE = -I. # place holder for now !TODO

all: OBJS

$(ESTIMATOR_MAINS): %: %.cpp $(OBJS)
	$(CC) $(INCLUDE) $(CFLAGS) $(OBJS) -o $@ $<

call_graph.o: call_graph.cpp call_graph.h trace.cpp trace.h span.cpp span.h
	$(CC) $(INCLUDE) $(CFLAGS) -c call_graph.cpp

trace.o: trace.cpp trace.h span.cpp span.h
	$(CC) $(INCLUDE) $(CFLAGS) -c trace.cpp

span.o: span.cpp span.h
	$(CC) $(INCLUDE) $(CFLAGS) -c span.cpp

clean:	
	rm -f *.o
