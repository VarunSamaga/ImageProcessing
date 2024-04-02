all: compile exec

compile:
	nvcc main.cu -o main -I/usr/local/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui

exec:
	./main input
	# ./main lowContrast