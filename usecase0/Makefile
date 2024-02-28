INCDIR      = /usr/local/include
LIBDIR      = /usr/local/lib

include make.def

SOURCES		= ../bgo/*.cpp main.cpp
OBJECTS	=	$(SOURCES:.cpp=.o)
TARGETS	=	main

all		: $(TARGETS)

../bgo/%.o	: ../bgo/%.cpp $(HEADERS)
		$(CXX) $(CXXFLAGS) -c $< -o $@

%.s 		: %.cpp $(HEADERS)
		$(CXX) -S $(CXXFLAGS) $<

%.o  		: %.cpp $(HEADERS)
		$(CXX) -c $(CXXFLAGS) $<

%  		: %.o
		$(CXX) $^ -o $@ $(LIB)

main: main.o ../bgo/find_max.o ../bgo/find_path.o
	$(CXX) $^ -o $@ $(LIB)

clean		:
		/bin/rm -rf a.out $(TARGETS) $(OBJECTS) *~

.SECONDARY	: $(OBJECTS)

-include $(SOURCES:%.cpp=%.d)
