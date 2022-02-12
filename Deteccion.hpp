
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

#include <opencv2/video/video.hpp>         // Procesamiento y escritura
#include <opencv2/videoio/videoio.hpp>     // de video
#include <opencv2/photo/photo.hpp>


//#include <opencv2/opencv.hpp> // Carga todas las librerías de OpenCV

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <string>
#include <fstream>
#include <iostream>
#include <sys/types.h>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <dirent.h>
#include <filesystem>
#include <opencv2/objdetect/objdetect.hpp> // Librería para detectar objetos 
#include <opencv2/features2d/features2d.hpp> // Librería que contiene los descriptores como SIFT
#include <opencv2/xfeatures2d/nonfree.hpp> // Contiene funciones que están protegidas y no son libres
#include <opencv2/ml/ml.hpp> // Paquete que contiene algoritmos de machine learning para ser usados en OpenCV
#include <opencv2/dnn/dnn.hpp> 
#include "Ejercicio/lbp-library-master/lbplibrary.hpp"
#include "Ejercicio/lbp-library-master/histogram.hpp"


using namespace std;
using std::sort;
namespace fs= std::filesystem;
using namespace cv; // Espacio de nombres de la librería de OpenCV
using namespace cv::ml;
using namespace lbplibrary;


#define compab_mask_inc(ptr,shift) \
{ value |= ((unsigned int)(cntr - *ptr) & 0x80000000) \
>> (31-shift); ptr++; }

class Deteccion{
    public:
        Deteccion();
        int* LBP8(const int*, int, int);
        vector<int> LBP8(Mat);
        vector<int> ELBP(Mat);
        vector<string> subirImagenes(string ruta);
        void creacionSVM(vector<string> imagenes);
        void creacionSVMELBP(vector<string> imagenes);
        void SVMPrediccion(string ruta, string svmFile);
        void SVMPrediccionELBP(string ruta, string svmFile);
        Mat SVMPrediccionCamara(Mat imagen, string svmFile);
        Mat SVMPrediccionCamaraDetec(Mat imagen, Mat recortar, string svmFile);
        void capturar(string svmFile);
        void capturarD(string svmFile);
};



