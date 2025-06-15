CC = gcc
CFLAGS = -I./include -Wall

SRC = $(wildcard src/*.c src/*/*.c)
OBJ = $(SRC:%.c=%.o)

main: $(OBJ)
	$(CC) $(CFLAGS) -o main $(OBJ) -lm

clean:
	rm -f $(OBJ) main