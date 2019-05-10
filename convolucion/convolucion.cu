#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


#define CHECK(call){const cudaError_t error = call;if (error != cudaSuccess){printf("Error: %s:%d, ", __FILE__, __LINE__);printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));exit(1);}}

// ------------------ cuda -----------------------------


// template to easily create new cuda arrays
template <class T>
T* cuda_array(int n){
  T* arr;
  size_t tam = n * sizeof(T);
  CHECK(cudaMalloc(&arr, tam));
  return arr;
}

// template to copy array from host to device
template <class T>
void cuda_H2D(T* h_arr, T* d_arr, int n){
  size_t tam = n * sizeof(T);
  CHECK(cudaMemcpy(d_arr, h_arr, tam, cudaMemcpyHostToDevice));
}

// template to copy array from device to host
template <class T>
void cuda_D2H(T* d_arr, T* h_arr, int n){
  size_t tam = n * sizeof(T);
  CHECK(cudaMemcpy(h_arr, d_arr, tam, cudaMemcpyDeviceToHost));
}


__global__ void conv(unsigned char* d_data, float* d_kernel, int k, int p, int pos_k, int pos_p, int h, int w){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if(i < w && j < h){
    float sum_c1 = 0;
    float sum_c2 = 0;
    float sum_c3 = 0;
    for (size_t r = 0; r < k; r++) {
      for (size_t c = 0; c < p; c++) {
        sum_c1 += d_data[(j - pos_k + r) * (3 * w) + ((i - pos_p + c) * 3)] * d_kernel[r * p + c];
        sum_c2 += d_data[(j - pos_k + r) * (3 * w) + ((i - pos_p + c) * 3) + 1] * d_kernel[r * p + c];
        sum_c3 += d_data[(j - pos_k + r) * (3 * w) + ((i - pos_p + c) * 3) + 2] * d_kernel[r * p + c];
      }
    }
    d_data[j * 3 * w + i * 3] = sum_c1;
    d_data[j * 3 * w + i * 3 + 1] = sum_c2;
    d_data[j * 3 * w + i * 3 + 2] = sum_c3;
  }
}

Mat aplicar_filtro(Mat& img_i, float* kernel, int k , int p, int i, int j){
  Mat img = img_i.clone();
  int h, w;
  h = img.size().height;
  w = img.size().width;


  float block_size = 16;
  dim3 block =  dim3(block_size, block_size, 1);
  dim3 grid =  dim3(ceil(w / block_size), ceil(h / block_size), 1);

  unsigned char* data = (unsigned char*)img.data;
  unsigned char* d_data = cuda_array<unsigned char>(h * w * 3);
  float* d_kernel = cuda_array<float>(k * p);


  cuda_H2D(data, d_data, h * w * 3);
  cuda_H2D(kernel, d_kernel, k * p);
  conv<<<grid, block>>>(d_data, d_kernel, k, p, i, j, h, w);
  cudaDeviceSynchronize();
  cuda_D2H(d_data, data, h * w * 3);
  cudaFree(d_kernel);
  return img;

}






int main(int argc, char const *argv[]) {
  Mat img = imread("img.jpg");
  resize(img, img,cv::Size(), 0.1, 0.1);




  float kernel_Sobel_X[9] =
  {-1, 0, 1,
  -2, 0, 2,
  -1, 0, 1};

  float kernel_Sobel_Y[9] =
  {-1, -2, -1,
   0, 0, 0,
   1, 2, 1};

  float kernel_perfilado[9] =
  {-1, -1, -1,
  -1, 9, -1,
  -1, -1, -1};

  float kernel_filtro_gaussiano_l[7] =
  {1.0 / 64.0,
  6.0 / 64.0,
  15.0 / 64.0,
  20.0 / 64.0,
  15.0 / 64.0,
  6.0 / 64.0,
  1.0 / 64.0};


  Mat sobel_x, sobel_y, perfilado, borde_suavizado, filtro_gaussiano;
  sobel_x = aplicar_filtro(img, &kernel_Sobel_X[0], 3, 3, 1, 1);
  sobel_y = aplicar_filtro(img, &kernel_Sobel_Y[0], 3, 3, 1, 1);
  perfilado = aplicar_filtro(img, &kernel_perfilado[0], 3, 3, 1, 1);
  filtro_gaussiano = aplicar_filtro(img, &kernel_filtro_gaussiano_l[0], 1, 7, 3, 0);
  filtro_gaussiano = aplicar_filtro(filtro_gaussiano, &kernel_filtro_gaussiano_l[0], 7, 1, 0, 3);

  imshow("Imagen original", img);
  imshow("Sobel x", sobel_x);
  imshow("Sobel y", sobel_y);
  imshow("Perfilado", perfilado);
  imshow("Filtro gaussiano", filtro_gaussiano);
  waitKey(0);


  return 0;
}
