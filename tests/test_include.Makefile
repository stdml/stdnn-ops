include_dir = $(CURDIR)/../include

srcs = $(wildcard $(include_dir)/nn/bits/ops/*.hpp)

objs = $(patsubst %.hpp,%.o,$(srcs))

kernels: \
	$(objs)

CFLAGS = \
	-I$(CURDIR)/../3rdparty/include \
	-I$(include_dir)

STD = -std=c++17

%.o: %.hpp
	$(CXX) $(STD) -c $(CFLAGS) $^ -o $@
