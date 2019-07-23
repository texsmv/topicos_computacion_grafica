
CXX		  := nvcc
CXX_FLAGS := -std=c++14

BIN		:= bin
SRC		:= src
INCLUDE	:= include
LIB		:= lib

LIBRARIES	:=  -ltgui -lsfml-graphics -lsfml-window -lsfml-system -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs 
EXECUTABLE	:= main


all: $(BIN)/$(EXECUTABLE)

run: clean all
	clear
	./$(BIN)/$(EXECUTABLE)

$(BIN)/$(EXECUTABLE): $(SRC)/*.cu
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) -L$(LIB) $^ -o $@ $(LIBRARIES)

clean:
	-rm $(BIN)/*


exe:
	./$(BIN)/$(EXECUTABLE)