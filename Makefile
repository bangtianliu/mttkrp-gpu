CC = nvcc
SOURCEDIR = ./

EXE = main

SOURCES = $(SOURCEDIR)/MTTKRP.cpp \
	      $(SOURCEDIR)/convert.cpp \
	      $(SOURCEDIR)/readtensor.cpp \
	      $(SOURCEDIR)/tensor.cpp

CU_SOURCES = $(SOURCEDIR)/gpuMTTKRP.cu	

IDIR = -I/usr/local/cuda/samples/common/inc -I/home/labuser/splatt-1.1.1/include

LDIR = -L/usr/local/cuda/lib64

H_FILES = $(wildcard *.h)
OBJS = $(SOURCES:.cpp=.o)

CU_OBJS=$(CU_SOURCES:.cu=.o)

DOUBLEFLAGS = -DDOUBLE
TYPEFLAGS = -DLONG
#typeflags = -DLONG -DINT -DSHORT -DCHAR

CFLAGS = -O3 -std=c++11  -Xcompiler -fopenmp $(DOUBLEFLAGS) $(TYPEFLAGS)

LFLAGS = -lm -lstdc++ -lsplatt 
SMS ?= 35 37 50 52 60
#SMS ?= 20 30 35 37 50 52 60

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

NVCCFLGAS = -O3 -std=c++11 $(DOUBLEFLAGS) $(TYPEFLAGS) -Xptxas -dlcm=ca -Xptxas -v

$(EXE) : $(OBJS) $(CU_OBJS)
	$(CC) $(CFLAGS) $(LFLAGS) $(GENCODE_FLAGS) -o $@ $?
$(SOURCEDIR)/%.o: $(SOURCEDIR)/%.cpp $(H_FILES)
	$(CC) $(CFLAGS) $(LFLAGS) $(IDIR) -c -o $@ $<
$(SOURCEDIR)/%.o: $(SOURCEDIR)/%.cu $(H_FILES)
	$(CC) $(NVCCFLGAS) $(LFLAGS) $(GENCODE_FLAGS) $(IDIR) -c -o $@ $<

clean:
	rm -f *.o main	
