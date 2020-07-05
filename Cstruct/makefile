all: main

CC = gcc -O3

lib: lib.c
	$(CC) $@ -lm
main: lib.o
	$(CC) $@ -lm

clean:
	rm -f main *.o

