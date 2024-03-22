INCDIR      = ../../include
LIBDIR      = ../../lib

include make.def

SOURCES		= ../bgo/*.cpp betweenness_centrality_benchmark.cpp sssp_benchmark.cpp
OBJECTS	=	$(SOURCES:.cpp=.o)
TARGETS	=	betweenness_centrality_benchmark sssp_benchmark

all		: $(TARGETS)

../bgo/%.o	: ../bgo/%.cpp $(HEADERS)
		$(CXX) $(CXXFLAGS) -c $< -o $@

%.s 		: %.cpp $(HEADERS)
		$(CXX) -S $(CXXFLAGS) $<

%.o  		: %.cpp $(HEADERS)
		$(CXX) -c $(CXXFLAGS) $<

%  		: %.o
		$(CXX) $^ -o $@ $(LIB)

betweenness_centrality_benchmark: betweenness_centrality_benchmark.o ../bgo/betweenness_centrality.o
	$(CXX) $^ -o $@ $(LIB)

sssp_benchmark: sssp_benchmark.o ../bgo/bfs.o ../bgo/delta_step.o ../bgo/dijkstra.o
	$(CXX) $^ -o $@ $(LIB)

clean		:
		/bin/rm -rf a.out $(TARGETS) $(OBJECTS) *~

.SECONDARY	: $(OBJECTS)

-include $(SOURCES:%.cpp=%.d)
