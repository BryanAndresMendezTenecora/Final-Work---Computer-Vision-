
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

// Librerías de OpenCV
#include <opencv2/core/core.hpp> // Contiene las definiciones de matrices y otras
// estructuras que permiten representar imágenes
#include <opencv2/highgui/highgui.hpp> // Contiene funciones para crear ventanas y otros componentes gráficos
#include <opencv2/imgproc/imgproc.hpp> // Contiene funciones para procesar imágenes
#include <opencv2/imgcodecs/imgcodecs.hpp> // Contiene funciones para poder leer imágenes en distintos formatos
#include <opencv2/video/video.hpp> // Contiene funciones para manipulación de vídeo
#include <opencv2/videoio/videoio.hpp> // Contiene funciones para lectura y escritura de videos

//#include <opencv2/opencv.hpp> // Carga todas las librerías de OpenCV

#include <opencv2/objdetect/objdetect.hpp>

using namespace std;

using namespace cv; // Espacio de nombres de la librería de OpenCV

/*
    Descriptor LBP basado en el paper Real-Time Surface Inspection by Texture
    de los autores Topi M¨aenp¨a¨a, Markus Turtinen and Matti Pietik¨ainen
*/

#define compab_mask_inc(ptr,shift) \
{ value |= ((unsigned int)(cntr - *ptr) & 0x80000000) \
>> (31-shift); ptr++; }

class LBPDescriptor{
    public:
        LBPDescriptor();
        int* LBP8(const int*, int, int);
        vector<int> LBP8(Mat);
};