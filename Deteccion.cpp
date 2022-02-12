
#include "Deteccion.hpp"



Deteccion::Deteccion(){
    cout << "Inicializa el descriptor ... " << endl;
}

/**
* Calculate a LBP8,1 feature vector for an image array.
* This function does not use interpolation. The input
* image is presented as a linear array, in raster-scan
* order. As a result, a newly allocated array of 256
* integers is returned.
**/
int* Deteccion::LBP8(const int* data, int rows, int columns){
    const int
    *p0 = data,
    *p1 = p0 + 1,
    *p2 = p1 + 1,
    *p3 = p2 + columns,
    *p4 = p3 + columns,
    *p5 = p4 - 1,
    *p6 = p5 - 1,
    *p7 = p6 - columns,
    *center = p7 + 1;
    int r,c,cntr;
    unsigned int value;
    int* result = (int*)malloc(256*sizeof(int));
    memset(result, 0, 256*sizeof(int));
    for (r=0;r<rows-2;r++){
        for (c=0;c<columns-2;c++){
            value = 0;
            cntr = *center - 1;
            compab_mask_inc(p0,0);
            compab_mask_inc(p1,1);
            compab_mask_inc(p2,2);
            compab_mask_inc(p3,3);
            compab_mask_inc(p4,4);
            compab_mask_inc(p5,5);
            compab_mask_inc(p6,6);
            compab_mask_inc(p7,7);
            center++;
            result[value]++;
        }
        p0 += 2;
        p1 += 2;
        p2 += 2;
        p3 += 2;
        p4 += 2;
        p5 += 2;
        p6 += 2;
        p7 += 2;
        center += 2;
    }
    return result;
}

vector<int> Deteccion::LBP8(Mat imagen){

    int *data = (int *) malloc(imagen.rows*imagen.cols*sizeof(int));
    for(int i=0,s=0;i<imagen.rows;i++){
        for(int j=0;j<imagen.cols;j++,s++){
            data[s] = (int) imagen.at<uchar>(i,j);
            //*(data+s) = (int) imagen.at<uchar>(i,j);
        }
    }

    int *descriptor = LBP8(data, imagen.rows, imagen.cols);
    vector<int> desc;
    for(int i=0;i<256;i++){
        desc.push_back(descriptor[i]);
    }

    free(data);

    return desc;
}

vector<string> Deteccion::subirImagenes(string ruta)
{
    vector<string> archivos;
    for(const auto &entry:fs::directory_iterator(ruta)){
        archivos.push_back(entry.path());
    }
    sort(archivos.begin(),archivos.end());
    return archivos;
}

vector<int> ELBPCL(Mat img_input)
{
	cv::Mat img_output;
	//cv::Mat img_input = cv::imread("Bryan-Mendez/n02091635_2.jpg");
    //cv::Mat img_input = cv::imread(ruta);
    //int width=120;
    //int height=120;
	//cv::imshow("Input", img_input);
	//cv::cvtColor(img_input, img_input, cv::COLOR_BGR2GRAY);
    //resize(img_input,img_input,Size(width,height));
    //Mat histograma= Mat::zeros(img_input.size(), img_input.type());

	//cv::GaussianBlur(img_input, img_input, cv::Size(7, 7), 5, 3, cv::BORDER_CONSTANT);

	LBP *lbp;
	//lbp = new OLBP;     // 0-255
	lbp = new ELBP;     // 0-255
	//lbp = new VARLBP;   // 0-953.0
	//lbp = new CSLBP;    // 0-15
	//lbp = new CSLDP;    // 0-15
	//lbp = new XCSLBP;   // 0-15
	//lbp = new SILTP;    // 0-80
	//lbp = new CSSILTP;  // 33-120
	//lbp = new SCSLBP;   // 0-15
	//lbp = new BGLBP;    // 0-239
	lbp->run(img_input, img_output);

	double min, max; cv::minMaxLoc(img_output, &min, &max);
    //std::cout << "min: " << min << ", max: " << max <<endl;
	cv::normalize(img_output, img_output, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    Mat histogramaLBP;
    //show_histogram("gray_hist_LBP",img_output);
    histogram(img_output, histogramaLBP,256);

    vector<int> histo;
    for(int i=0; i<histogramaLBP.cols;i++){
        //cout << histograma[i] << endl;
        int pixel=(int) histogramaLBP.at<uchar>(0,i);
        //cout << pixel << ", " ;
        histo.push_back(pixel);
    }
    //cout << endl;
	//cv::imshow("Output", img_output);
    delete lbp;
    return histo;
}

void Deteccion::creacionSVMELBP(vector<string> imagenes){

    float clases[5335];

    for(int i=0; i<5335; i++){
        if(i<1886){
            clases[i]=1;
        }else{
            clases[i]=-1;
        }
    }

    float puntos[5335][256];
    for(int i=0; i<imagenes.size();i++){
        Mat imagen= imread(imagenes[i],IMREAD_GRAYSCALE);
        vector<int> descELBP=ELBPCL(imagen);
        for(int j=0;j<256;j++){
            puntos[i][j]=descELBP[j];
        }
    }

    Mat matrizPuntos(5335, 256, CV_32F, puntos);
    Mat matrizClases(5335, 1, CV_32SC1, clases);


    // Creamos la máquina de soporte vectorial y la entrenamos
    Ptr<SVM> svm = SVM::create();
    // Indicamos el tipo de SVM, es decir, si se usará para clasificar 2 clases o más
    svm->setType(SVM::C_SVC); // Con este tipo de SVM podemos clasificar más de 2 clases
    svm->setKernel(SVM::DEGREE); // Se especifica el tipo de Kernel (función para definir zonas de clasificación)

    // Especificamos los criterios de entrenamiento (número de iteraciones, error, etc.)
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-7));
    svm->train(matrizPuntos, ROW_SAMPLE, matrizClases);
    svm->save("svmELBP.xml");
}

void Deteccion::creacionSVM(vector<string> imagenes){

    float clases[5335];

    for(int i=0; i<5335; i++){
        if(i<1886){
            clases[i]=1;
        }else{
            clases[i]=-1;
        }
    }

    float puntos[5335][256];
    for(int i=0; i<imagenes.size();i++){
        Mat imagen= imread(imagenes[i],IMREAD_GRAYSCALE);
        vector<int> descLBP=LBP8(imagen);
        for(int j=0;j<256;j++){
            puntos[i][j]=descLBP[j];
        }
    }

    Mat matrizPuntos(5335, 256, CV_32F, puntos);
    Mat matrizClases(5335, 1, CV_32SC1, clases);

    // Creamos la máquina de soporte vectorial y la entrenamos
    Ptr<SVM> svm = SVM::create();
    // Indicamos el tipo de SVM, es decir, si se usará para clasificar 2 clases o más
    svm->setType(SVM::C_SVC); // Con este tipo de SVM podemos clasificar más de 2 clases
    svm->setKernel(SVM::DEGREE); // Se especifica el tipo de Kernel (función para definir zonas de clasificación)
    // Especificamos los criterios de entrenamiento (número de iteraciones, error, etc.)
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-7));
    svm->train(matrizPuntos, ROW_SAMPLE, matrizClases);
    svm->save("svm.xml");

}




void Deteccion::SVMPrediccion(string imagen, string svmFile){

    Ptr<SVM> svm = SVM::create();
    svm=svm->load(svmFile);
    cout <<"Se cargo el SVM" << endl;

    int width=120;
    int height=120;
    Mat img=imread(imagen, IMREAD_GRAYSCALE);
    resize(img,img,Size(width,height));
    Mat imgColor=imread(imagen);
    
    vector<int> desc=LBP8(img);
    
    float puntos[1][256];

    for(int i=0; i<256;i++){
        puntos[0][i]=desc[i];
    }

    Mat imgData(1,256, CV_32F, puntos);

    float r =svm->predict(imgData);
    
    cout << "Prediccion: " << r << endl;
    cout << "Paso" << endl;

    if(r<0.0){
        putText(imgColor, "Usa Lentes", Point(30,30), FONT_HERSHEY_PLAIN, 2, Scalar(255,10,10),2);
    }else if(r>0.0){
        putText(imgColor, "NO usa Lentes", Point(30,30), FONT_HERSHEY_PLAIN, 2, Scalar(10,10,255),2);
    }
    imshow("Img",img);
    imshow("Img Color", imgColor);
    waitKey(0);



}

void Deteccion::SVMPrediccionELBP(string imagen, string svmFile){

    Ptr<SVM> svm = SVM::create();
    svm=svm->load(svmFile);
    cout <<"Se cargo el SVM" << endl;

    int width=120;
    int height=120;
    Mat img=imread(imagen, IMREAD_GRAYSCALE);
    //Mat recortada=img[10:650,300:400];
    //Rect r(100,100,100,100);
    //Mat h
    //Mat imgCut=img(Rect(200,400,400,400));


    resize(img,img,Size(width,height));
    Mat imgColor=imread(imagen);
    
    vector<int> desc=ELBPCL(img);
    
    float puntos[1][256];

    for(int i=0; i<256;i++){
        puntos[0][i]=desc[i];
    }

    Mat imgData(1,256, CV_32F, puntos);

    float r =svm->predict(imgData);
    
    cout << "Prediccion: " << r << endl;
    //cout << "Paso" << endl;

    if(r>0.0){
        putText(imgColor, "Usa Lentes", Point(30,30), FONT_HERSHEY_PLAIN, 2, Scalar(255,10,10),2);
    }else if(r<0.0){
        putText(imgColor, "NO usa Lentes", Point(30,30), FONT_HERSHEY_PLAIN, 2, Scalar(10,10,255),2);
    }
    imshow("Img",img);
    imshow("Img Color", imgColor);
    //imshow("REcortada",imgCut);
    waitKey(0);



}

Mat Deteccion::SVMPrediccionCamara(Mat imagen, string svmFile){
    Mat imgColor=imagen.clone();
    Ptr<SVM> svm = SVM::create();
    svm=svm->load(svmFile);
    cout <<"Se cargo el SVM" << endl;

    int width=120;
    int height=120;
    cvtColor(imagen,imagen,COLOR_BGR2GRAY);
    //Mat img=imread(imagen, IMREAD_GRAYSCALE);
    resize(imagen,imagen,Size(width,height));
    
    
    vector<int> desc=ELBPCL(imagen);
    
    float puntos[1][256];

    for(int i=0; i<256;i++){
        puntos[0][i]=desc[i];
    }

    Mat imgData(1,256, CV_32F, puntos);

    float r =svm->predict(imgData);
    
    cout << "Prediccion: " << r << endl;
    cout << "Paso" << endl;

    if(r<0.0){
        putText(imgColor, "Usa Lentes", Point(30,30), FONT_HERSHEY_PLAIN, 2, Scalar(255,10,10),2);
    }else if(r>0.0){
        putText(imgColor, "NO usa Lentes", Point(30,30), FONT_HERSHEY_PLAIN, 2, Scalar(10,10,255),2);
    }
    //imshow("Img",img);
    //imshow("Img Color", imgColor);
    //waitKey(0);

    return imgColor;

}


Mat Deteccion::SVMPrediccionCamaraDetec(Mat imagen, Mat recortar, string svmFile){
    Mat imgColor=imagen.clone();
    Ptr<SVM> svm = SVM::create();
    svm=svm->load(svmFile);
    cout <<"Se cargo el SVM" << endl;

    int width=120;
    int height=120;
    cvtColor(recortar,recortar,COLOR_BGR2GRAY);
    //Mat img=imread(imagen, IMREAD_GRAYSCALE);
    //resize(recortar,recortar,Size(width,height));
    
    
    vector<int> desc=ELBPCL(recortar);
    
    float puntos[1][256];

    for(int i=0; i<256;i++){
        puntos[0][i]=desc[i];
    }

    Mat imgData(1,256, CV_32F, puntos);

    float r =svm->predict(imgData);
    
    cout << "Prediccion: " << r << endl;
    cout << "Paso" << endl;

    if(r<0.0){
        putText(imgColor, "Usa Lentes", Point(30,30), FONT_HERSHEY_PLAIN, 1, Scalar(255,10,10),2);
    }else if(r>0.0){
        putText(imgColor, "NO usa Lentes", Point(30,30), FONT_HERSHEY_PLAIN, 1, Scalar(10,10,255),2);
    }
    //imshow("Img",img);
    //imshow("Img Color", imgColor);
    //waitKey(0);

    return imgColor;

}

void Deteccion::capturar(string svmFile){
   // Mat copia=fondo.clone();

    VideoCapture video(0);
    //VideoCapture video1(this->getVideo());
    //codigo=VideoWriter_fourcc();  
    //int numero = CAP_PROP_FOURCC(4);
    int fourcc=VideoWriter::fourcc('M','J','P','G');
    VideoWriter salida("VideoGrabado.avi",fourcc,20,Size(640,480));
    Mat frame;
    Mat corto;
    Mat r;
    Mat finally;
     if(video.isOpened()){
        namedWindow("Video1", WINDOW_AUTOSIZE);
        while(3==3){
            video >> frame;
            //video1 >> corto;
            /*
            if(frame.empty()){
                cout << "Video Finalizado" << endl;
                break;
            }*/
            frame=SVMPrediccionCamara(frame,svmFile);
            imshow("Video1", frame);
            cout<<frame.cols << " " << frame.rows << endl;
            salida.write(frame);

            if(waitKey(23)==27){
                break;
            }
        }
        video.release();
        salida.release();
    }
}

void Deteccion::capturarD(string svmFile){
     Mat rostros1;// = imread("Rostros1.jpg");
    //Mat rostros2 = imread("Rostros2.jpg");
    VideoCapture capture(0);
    //resize(rostros1, rostros1, Size(), 0.75, 0.75);
    //resize(rostros2, rostros2, Size(), 0.5, 0.5);

    // Sugerencia
    // Pruebe escalando la imagen a 300x300, que es la sugerencia que dan los autores de la red neuronal

    // Ejemplo de cómo cargar una Red Neuronal entrenada en TensorFlow
    string configuracion = "red-neuronal/opencv_face_detector.pbtxt.txt"; // Archivo de configuraicón de la Red Neuronal
    string parametros = "red-neuronal/opencv_face_detector_uint8.pb"; // Archivo de parámetros de la red neuronal
    
    cout << "Intenta cargar el modelo de red neuronal" << endl;

    dnn::Net modelo = dnn::readNetFromTensorflow(parametros, configuracion);

    cout << "Carga el modelo con éxito" << endl;

    // Opciones cuando se tiene tarjeta NVidia para procesamiento paralelo
    // modelo.setPreferableBackend(DNN_BACKEND_CUDA);
    // modelo.setPreferableTarget(DNN_TARGET_CUDA);

    // Ingreso de la imagen en la red neuronal
    
    for(;;){
        capture >> rostros1;
        Mat ros=rostros1.clone();
        resize(rostros1,rostros1,Size(300,300),0.5,0.5);

        Mat blob = dnn::blobFromImage(rostros1, 1.0, rostros1.size(), Scalar(104, 177, 123),true, false);

        // Se ingresa la imagen en la red neuronal
        vector<string> nombresCapas = modelo.getLayerNames();
        /*for(string nombre:nombresCapas){
            cout << "Nombre. " << nombre << endl;
        }*/

        modelo.setInput(blob, "data");
        Mat deteccion = modelo.forward("detection_out");

        // Obtenemos una matriz con los resultados de detección
        Mat resultadosDeteccion(deteccion.size[2],deteccion.size[3],CV_32F,deteccion.ptr<float>());

        // Obtenemos cuántas caras se detectaron
        //cout << "Caras detectadas por el modelo: " << resultadosDeteccion.rows << endl;

        int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        float umbralConfianza = 0.0;
        Mat imgCut;

        for (int i=0;i<resultadosDeteccion.rows;i++){
            umbralConfianza = resultadosDeteccion.at<float>(i,2);
            if(umbralConfianza>0.9){
                //cout << "Cara umbral mayor " << i << endl;

                x1 = static_cast<int> (resultadosDeteccion.at<float>(i, 3)*rostros1.cols);
                y1 = static_cast<int> (resultadosDeteccion.at<float>(i, 4)*rostros1.rows);
                x2 = static_cast<int> (resultadosDeteccion.at<float>(i, 5)*rostros1.cols);
                y2 = static_cast<int> (resultadosDeteccion.at<float>(i, 6)*rostros1.rows);

               //rectangle(rostros1, Point(x1,y1), Point(x2,y2), Scalar(10,200,200), 2);
            }
        }
        
        imgCut=ros(Rect(x1+90,y1,x2+50,y2+50));
        //imwrite("cut5.jpg",imgCut);
        //namedWindow("Cortado",WINDOW_AUTOSIZE);
        //imshow("Cortado",imgCut);
        //waitKey(0);
        //namedWindow("Rostros1", WINDOW_AUTOSIZE);
        //imshow("Rostros1", rostros1);
        //Mat cl=rostros1.clone()
        rostros1=SVMPrediccionCamaraDetec(rostros1,imgCut,svmFile);
        imshow("Video1", rostros1);
        
        if(waitKey(33)==27){
            break;
        }

    }
    // namedWindow("Rostros1", WINDOW_AUTOSIZE);
    // namedWindow("Rostros2", WINDOW_AUTOSIZE);
    // imshow("Rostros1", rostros1);
    // imshow("Rostros2", rostros2);

    //waitKey(0);

    destroyAllWindows();

    //return 0;
}
