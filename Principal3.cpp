#include "Deteccion.hpp"

string formato(int numero ){
    if(numero < 10){
        string n=to_string(numero);
        string r="dataset/image000"+n+".jpg";
        return r;
    }else if(numero < 100){
        string n=to_string(numero);
        string r="dataset/image00"+n+".jpg";
        return r;
    }else if(numero < 1000){
        string n=to_string(numero);
        string r="dataset/image0"+n+".jpg";
        return r;
    }else{
        string n=to_string(numero);
        string r="dataset/image"+n+".jpg";
        return r;
    }
}

void cambiar(vector<string> ruta){
    int cont=250;
    for(int i=0;i<1;i++){
        for(int j=0;j<ruta.size();j++){
            Mat img=imread(ruta[j]);
            resize(img,img,Size(120,120));
            imwrite("SinLenteR/img"+to_string(cont)+".jpg",img);
            cont++;
        }
    }
}

int main(int argc, char *argv[]){

    Deteccion *d=new Deteccion();
    
    string ruta1="data/Glass";
    string ruta2="data/NoGlass";
    //vector<string> imag1=d->subirImagenes(ruta1);
    //vector<string> imag2=d->subirImagenes(ruta2);
    //cout << "Ruta1: " << imag1.size() << " Ruta2:" << imag2.size() << endl;

     /*-- Cambiar nombre IMAGENES;
    int cont=1;
    for(int i=0; i<imag1.size();i++){
        Mat imagen=imread(imag1[i]);
        string n=to_string(cont);
        //string rutaG="imagenes/image"+n+".jpg";
        string rutaG=formato(cont);
        //cout << rutaG << endl;
        imwrite(rutaG,imagen);
        cont=cont+1;
    }
    for(int i=0; i<imag2.size();i++){
        Mat imagen=imread(imag2[i]);
        string n=to_string(cont);
        //string rutaG="imagenes/image"+n+".jpg";
        string rutaG=formato(cont);
        //cout << rutaG << endl;
        imwrite(rutaG,imagen);
        cont=cont+1;
    }
    */
    string ruta="dataset";
    vector<string> imagenes=d->subirImagenes(ruta);
    //cambiar(imagenes);mak
    //cout << imagenes.size() <<endl;
    //d->creacionSVMELBP(imagenes);
    //d->creacionSVM(imagenes);
    //string svmRuta="svm.xml";
    string svmRuta="svmELBP.xml";
   //string imgPrediccion="imagenes/image0753.jpg";
    string imgPrediccion="lentes.jpg";
    //d->SVMPrediccion(imgPrediccion,svmRuta);
    //d->SVMPrediccionELBP(imgPrediccion,svmRuta);
    //d->capturar(svmRuta);
    d->capturarD(svmRuta);
   
    delete d;
    return 0;


}