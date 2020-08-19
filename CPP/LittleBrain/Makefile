CC=gcc
CPP=g++
CFLAGS = -I. -g
TARGET = main

SRCS = \
		LittleBrain/main.cpp \
		LittleBrain/Types/*.cpp \
		LittleBrain/Neural_Network/*.cpp \
		LittleBrain/Activation/*.cpp 

OBJS = $(SRCS: .cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
		$(CPP) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(OBJS)

.c.o:
		$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@
.cpp.o:
		$(CPP) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	rm *.o $(TARGET)