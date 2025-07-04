CC = gcc
CFLAGS = -I./include -I./src -Wall

SRC = $(wildcard src/*.c src/*/*.c)
OBJ = $(patsubst src/%.c, build/%.o, $(SRC))

main: $(OBJ)
	$(CC) $(CFLAGS) -o main $(OBJ) -lm

build/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f main
	rm -rf build