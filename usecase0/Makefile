PROJECT_ROOT	= ${CURDIR}

INCDIR      = ./include
LIBDIR      = ./lib

include make.def

SOURCES		= main.c

OBJECTS		= $(SOURCES:.c=.o)
TARGETS		= $(SOURCES:.c=)
HEADERS		= $(wildcard $(INCDIR)/*.h)

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
