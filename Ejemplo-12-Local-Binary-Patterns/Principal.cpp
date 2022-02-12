
#include "LBPDescriptor.hpp"


int main(int argc, char *argv[]){

    Mat textura = imread("Textura.png", IMREAD_GRAYSCALE);

    LBPDescriptor *lbp = new LBPDescriptor();
    vector<int> desc = lbp->LBP8(textura);

    for(int i=0;i<desc.size();i++){
        cout << desc[i] << "|";
    }
    cout << endl;

    namedWindow("Textura", WINDOW_AUTOSIZE);
    imshow("Textura", textura);

    waitKey(0);

    destroyAllWindows();

    return 0;
}