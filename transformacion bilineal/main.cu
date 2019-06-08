#include <iostream>
#include <vector>
#include <tuple>

#include <SFML/Graphics.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/videoio/videoio.hpp"

#include "cud_defs.h"

using namespace cv;
using namespace std;


__device__ void d_apply_bilineal(float* coeff_x, float* coeff_y, int x_i, int y_i, int& x_o, int& y_o){
  x_o = coeff_x[0] * x_i + coeff_x[1] * y_i + coeff_x[2] * (x_i * y_i) + coeff_x[3];
  y_o = coeff_y[0] * x_i + coeff_y[1] * y_i + coeff_y[2] * (x_i * y_i) + coeff_y[3];
}
__global__ void apply_bilineal_mat(uchar* mat_i, uchar* mat_o, float* d_coeff_x, float* d_coeff_y, int h, int w, int h_o, int w_o){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < h && j < w) {
    int i_o, j_o;
    d_apply_bilineal(d_coeff_x, d_coeff_y, j, i, j_o, i_o);
    mat_o[i_o * w_o * 3 + j_o * 3] = mat_i[i * w * 3 + j * 3];
    mat_o[i_o * w_o * 3 + j_o * 3 + 1] = mat_i[i * w * 3 + j * 3 + 1];
    mat_o[i_o * w_o * 3 + j_o * 3 + 2] = mat_i[i * w * 3 + j * 3 + 2];

    // mat_o[i * w_o * 3 + j * 3] = mat_i[i * w * 3 + j * 3];
    // mat_o[i * w_o * 3 + j * 3 + 1] = mat_i[i * w * 3 + j * 3 + 1];
    // mat_o[i * w_o * 3 + j * 3 + 2] = mat_i[i * w * 3 + j * 3 + 2];

  }
}



void apply_bilineal(float* coeff_x, float* coeff_y, int x_i, int y_i, int& x_o, int& y_o);
void get_mat_bilineal(vector<pair<float, float> > input_points, vector<pair<float, float> > output_points, vector<float>& coeff_x, vector<float>& coeff_y);

int main()
{
  sf::RenderWindow window(sf::VideoMode(1200, 900), "RenderWindow");
  sf::Image image;
  sf::Texture texture;
  sf::Event event;
  sf::Sprite sprite;
  cv::Mat frameRGB, frameRGBA;

  sf::Image image2;
  sf::Texture texture2;
  sf::Sprite sprite2;
  cv::Mat res2;


  frameRGB = imread("stark.jpeg");

  vector<pair<float, float> > input_points = {make_pair(0, 0), make_pair(0, 5), make_pair(5, 0), make_pair(5, 5)};
  vector<pair<float, float> > output_points = {make_pair(5, 5), make_pair(10, 10), make_pair(10, 5), make_pair(15, 10)};

  // vector<pair<float, float> > input_points = {make_pair(0, 0), make_pair(5, 0), make_pair(0, 5), make_pair(5, 5)};
  // vector<pair<float, float> > output_points = {make_pair(5, 5), make_pair(10, 10), make_pair(5, 10), make_pair(10, 15)};

  vector<float> coeff_x, coeff_y;
  get_mat_bilineal(input_points, output_points, coeff_x, coeff_y);

  int x_min, x_max, y_min, y_max;
  x_min = 0; x_max = 0; y_min = 0; y_max = 0;
  int x_o, y_o;

  vector<pair<float, float> > esquinas = {make_pair(0, 0), make_pair(frameRGB.cols - 1, 0), make_pair(0, frameRGB.rows - 1), make_pair(frameRGB.cols - 1, frameRGB.rows - 1)};
  for (size_t i = 0; i < esquinas.size(); i++) {
    apply_bilineal(coeff_x.data(), coeff_y.data(), esquinas[i].first, esquinas[i].second, x_o, y_o);
    if(x_o > x_max)
      x_max = x_o;
    if(x_o < x_min)
      x_min = x_o;
    if(y_o > y_max)
      y_max = y_o;
    if(y_o < y_min)
      y_min = y_o;

  }
  cout<<x_min<<" - "<<x_max<<endl;
  cout<<y_min<<" - "<<y_max<<endl;

  Mat res = Mat::zeros(y_max - y_min + 1, x_max - x_min + 1, CV_8UC3);
  // Mat res = Mat::zeros(frameRGB.rows, frameRGB.cols, CV_8UC3);

  cout<<frameRGB.cols<<" - "<<frameRGB.rows<<endl;
  cout<<res.cols<<" - "<<res.rows<<endl;


  float block_size = 16;
  dim3 block =  dim3(block_size, block_size , 1);
  dim3 grid =  dim3(ceil(frameRGB.rows / block_size), ceil(frameRGB.cols / block_size), 1);

  uchar* d_mat_i = cuda_array<uchar>(frameRGB.cols * frameRGB.rows * 3);
  cuda_H2D<uchar>(frameRGB.data, d_mat_i, frameRGB.cols * frameRGB.rows * 3);

  CHECK(cudaDeviceSynchronize());

  uchar* d_mat_o = cuda_array<uchar>(res.cols * res.rows * 3);

  cuda_H2D<uchar>(res.data, d_mat_o, res.cols * res.rows * 3);

  CHECK(cudaDeviceSynchronize());


  float *d_coeff_x, *d_coeff_y;
  d_coeff_x = cuda_array<float>(coeff_x.size());
  d_coeff_y = cuda_array<float>(coeff_y.size());

  cuda_H2D<float>(coeff_x.data(), d_coeff_x, coeff_x.size());
  cuda_H2D<float>(coeff_y.data(), d_coeff_y, coeff_y.size());


  apply_bilineal_mat<<<grid, block>>>(d_mat_i, d_mat_o, d_coeff_x, d_coeff_y, frameRGB.rows, frameRGB.cols, res.rows, res.cols);
  CHECK(cudaDeviceSynchronize());

  cuda_D2H(d_mat_o, res.data, res.cols * res.rows * 3);
  CHECK(cudaDeviceSynchronize());

  // imshow("orig",frameRGB);
  // imshow("orig_m",res);
  // waitKey(0);





  cv::cvtColor(frameRGB, frameRGBA, cv::COLOR_BGR2RGBA);
  image.create(frameRGBA.cols, frameRGBA.rows, frameRGBA.ptr());
  texture.loadFromImage(image);
  sprite.setTexture(texture);

  cv::cvtColor(res, res2, cv::COLOR_BGR2RGBA);
  image2.create(res2.cols, res2.rows, res2.ptr());
  texture2.loadFromImage(image2);
  sprite2.setTexture(texture2);

  sprite2.setPosition(0,frameRGB.rows);

  while (window.isOpen())  {
    // cap >> frameRGB;

    if(frameRGB.empty())    {
      break;
    }




    while (window.pollEvent(event))    {
      if (event.type == sf::Event::Closed)
        window.close();
    }

    window.draw(sprite2);
    window.draw(sprite);
    window.display();
  }
  return 0;


}
