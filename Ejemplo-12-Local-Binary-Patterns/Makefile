all:
	g++ Principal.cpp LBPDescriptor.cpp --std=c++17 -I/home/vlarobbyk/aplicaciones/Librerias/opencv/opencvi/include/opencv4/ -L/home/vlarobbyk/aplicaciones/Librerias/opencv/opencvi/lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_video -lopencv_videoio -lopencv_objdetect -o vision.bin

run:
	./vision.bin

saludo:
	echo "Hola Mundo desde C++!"