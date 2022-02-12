

#include "LBPDescriptor.hpp"

LBPDescriptor::LBPDescriptor(){
    cout << "Inicializa el descriptor ... " << endl;
}

/**
* Calculate a LBP8,1 feature vector for an image array.
* This function does not use interpolation. The input
* image is presented as a linear array, in raster-scan
* order. As a result, a newly allocated array of 256
* integers is returned.
**/
int* LBPDescriptor::LBP8(const int* data, int rows, int columns){
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

vector<int> LBPDescriptor::LBP8(Mat imagen){

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