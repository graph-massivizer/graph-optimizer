INCDIR      = /usr/local/include
LIBDIR      = /usr/local/lib

include make.def

SOURCES		= main.c

OBJECTS		= $(SOURCES:.c=.o)
TARGETS		= $(SOURCES:.c=)

all		: $(TARGETS)

%.s 		: %.c $(HEADERS)
		$(CC) -S $(CCFLAGS) $<

%.o  		: %.c $(HEADERS)
		$(CC) -c $(CCFLAGS) $<

%  		: %.o
		$(CC) $^ -o $@ $(LIB)

clean		:
		/bin/rm -rf a.out $(TARGETS) $(OBJECTS) *~

.SECONDARY	: $(OBJECTS)

-include $(SOURCES:%.c=%.d)
