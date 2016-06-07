################################################################
#
# Makefile for the Nearest Neighbor Library for Motion Planning
#
# This is the non-configurable part of the Makefile.
# See the file 'Makefile.config' for configuration switches.
#
#
################################################################

# configurable part of the Makefile
include ../Makefile.config

######## INC configuration

INC = $(OTHERINC)

######## LIB configuration

LIB = $(OTHERLIB)

######## Flags

FLAGS = $(C++FLAGS) -pg

DEPSRC = *.C


#############
SOURCES = ANN.C kd_tree.C kd_util.C kd_split.C kd_search.C \
	nn.C multiann.C

HEADERS = ANN.h kd_tree.h kd_util.h kd_split.h \
	kd_search.h nn.h multiann.h

OBJECTS = $(SOURCES:.C=.o)

#-----------------------------------------------------------------------------
# Rules
#-----------------------------------------------------------------------------

.C.o:
	$(C++) $(FLAGS) -c -Wno-deprecated $< $(INC)

#-----------------------------------------------------------------------------
# Make the library
#-----------------------------------------------------------------------------

$(LIBDIR)/$(ANNLIB): $(OBJECTS)
	$(MAKELIB) $(ANNLIB) $(OBJECTS)
	$(RANLIB) $(ANNLIB)
	mv $(ANNLIB) $(LIBDIR)

all: libDNN.a

libDNN.a: $(OBJECTS)
	ar ruv ../lib/libDNN.a $(OBJECTS)

#-----------------------------------------------------------------------------
# Cleaning
#-----------------------------------------------------------------------------

clean:
	-rm -f *.o core

realclean: clean

backup:
	@echo "Creating tar archive..."
	@echo "-----------------------"
	tar -cvf backup_src.tar *.[Ch] Makefile*
	gzip backup_src.tar 
	@echo "DONE!"
count:
	@wc *.[Ch]

# DO NOT DELETE
