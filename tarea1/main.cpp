#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;
unsigned char* readBMP(char* filename)
{
    int i;
    FILE* f = fopen(filename, "rb");
    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f);

    int width = *(int*)&info[18];
    int height = *(int*)&info[22];

    short n_bits = *(short*)&info[28];

    cout<<width<<endl;
    cout<<height<<endl;
    cout<<n_bits<<endl;

    int size = 3 * width * height;
    unsigned char* data = new unsigned char[size];
    fread(data, sizeof(unsigned char), size, f);
    fclose(f);
    unsigned char* n_data = new unsigned char[size];

    int ind;
    for(int i = 0; i < height; i ++){
      for(int j = 0; j < width * 3; j ++){
        n_data[i * width * 3 + j] = data[(height - 1 - i) * width * 3 + j];
      }
        // cout<< i<<endl;
    }

    Mat img_n(height, width, CV_8UC3, n_data);
    // Mat img(height, width, CV_8UC3, data);
    imshow("img_n", img_n);
    waitKey(0);
    // imshow("img", img);

    return data;
}

unsigned char* readBMP_8(char* filename)
{
    int i;
    FILE* f = fopen(filename, "rb");
    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f);

    int width = *(int*)&info[18];
    int height = *(int*)&info[22];

    short n_bits = *(short*)&info[28];

    cout<<width<<endl;
    cout<<height<<endl;
    cout<<n_bits<<endl;

    int size =  width * height;
    unsigned char* data = new unsigned char[size];
    fread(data, sizeof(unsigned char), size, f);
    fclose(f);
    unsigned char* n_data = new unsigned char[size * 3];

    int ind;
    for(int i = 0; i < height; i ++){
      for(int j = 0; j < width ; j ++){
        int Red, Green, Blue;
        int Color = (int)data[(height - 1 - i) * width + j];

        Red   = (int)(Color >> 5) * 255 / 7;
        Green = (int)((Color >> 2) & 0x07) * 255 / 7;
        Blue  = (int)(Color & 0x03) * 255 / 3;

        n_data[i * width * 3 + (j * 3)] = (uchar)Blue;
        n_data[i * width * 3 + (j * 3) + 1] = (uchar)Green;
        n_data[i * width * 3 + (j * 3) + 2] = (uchar)Red;
      }
        // cout<< i<<endl;
    }


    Mat img_n(height, width, CV_8UC3,   n_data);
    // Mat img(height, width, CV_8UC3, data);
    imshow("img_n", img_n);
    waitKey(0);
    // imshow("img", img);

    return data;
}

int main(int argc, char const *argv[]) {

  cout<<"Lectura de imagen de 24 bits"<<endl;
  readBMP("t1_24b.bmp");

  cout<<"Lectura de imagen de 8 bits"<<endl;
  readBMP_8("t1_8b.bmp");
  return 0;
}
