install eigen library
apt-get install libeigen3-dev

install sfml library
apt-get install libsfml-dev

install cuda toolkit


Actualmente solo se implemento la transformacion bilineal, falta pedir el punto para realizar la transformacion en la imagen


ejecutar
g++ -c -l. bilineal.cpp -std=c++11 `pkg-config --cflags eigen3` -o bilineal.cpp.o

nvcc main.cu bilineal.cpp.o -std=c++14 -lsfml-graphics -lsfml-window -lsfml-system -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -o exe

./exe
