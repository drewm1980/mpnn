################################################################
#
# Makefile for the Nearest Neighbor Library for Motion Planning
#
# This is the non-configurable part of the Makefile.
# See the file 'Makefile.config' for configuration switches.
#
#
################################################################

OS = Linux

C++ = g++32

C++FLAGS = -g -Wall
LDFLAGS = 
CPPFLAGS = 

########   ANN Configuration
DNNINC = -I/home/lavalle/yershova/MPNN/include
DNNLIB = -L/home/lavalle/yershova/MPNN/lib -lDNN
DNNFLAGS = -DUSE_DNN

######## INC configuration

INC = $(CPPFLAGS) $(DNNINC)

######## LIB configuration

LIB = $(DNNLIB)

######## Flags

DEPFLAGS = $(DNNFLAGS)
FLAGS = $(C++FLAGS) $(DEPFLAGS)

DEPSRC = *.C

####### How to make things

.C.o:
	$(C++) $(FLAGS) -c $< $(INC)


######## All executables

PLANGL = test

all: $(PLANGL)


test:
	$(C++) $(FLAGS) -Wno-deprecated test.C -o test $(INC) $(LIB)

######## other commands

realclean:
	rm -f *.o test

clean:
	rm -f *.o test

# DO NOT DELETE

