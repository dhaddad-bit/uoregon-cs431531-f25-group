cc = gcc

sources = test_csr mmio
objects = $(addsuffix .o, $(basename $(sources)))

flags = -g -std=c11

target = test_csr

all: $(target)

make_b: make_b.o 
	$(CC) $(flags) -o make_b make_b.o

$(target) : $(objects)
	$(cc) $(flags) -o $(target) $(objects)

%.o : %.c
	$(cc) -c $(flags) $< -o $@


clean:
	rm -rf $(target) make_b *.o