#ifndef UTILS_H
#define UTILS_H

#include "stdio.h"
#include "stdlib.h"
#include <ctime>

#include <SFML/Graphics.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/videoio/videoio.hpp"

#define at2(m, j, i, w) (m[(j) * (w) + (i)])
#define at3(m, j, i, k, w, s) (m[((j) * (w) + (i)) * (s) + (k)])
#define at4(m, j, i, k, l, w, s, t) (m[(((j) * (w) + (i)) * (s) + (k)) * (t) + (l)])
#define CHECK(call){const cudaError_t error = call;if (error != cudaSuccess){printf("Error: %s:%d, ", __FILE__, __LINE__);printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));exit(1);}}
#define devs() CHECK(cudaDeviceSynchronize())



namespace cudu
{
    /* ------------------ cuda utils----------------------------- */ 

    

    /* template to easily create cuda arrays */ 
    template <class T>
    T* array(unsigned int n){
        T* arr;
        size_t tam = n * sizeof(T);
        CHECK(cudaMalloc(&arr, tam));
        return arr;
    }

    /* template to copy array from host to device */ 
    template <class T>
    void h2d(T*& h_arr, T*& d_arr, unsigned int n){
        size_t tam = n * sizeof(T);
        CHECK(cudaMemcpy(d_arr, h_arr, tam, cudaMemcpyHostToDevice));
    }

    /* template to copy array from device to host */
    template <class T>
    void d2h(T*& d_arr, T*& h_arr, unsigned int n){
        size_t tam = n * sizeof(T);
        CHECK(cudaMemcpy(h_arr, d_arr, tam, cudaMemcpyDeviceToHost));
    }

    /* template to copy array from device to device */ 
    template <class T>
    void d2d(T*& d_arr, T*& d_arr2, unsigned int n){
        size_t tam = n * sizeof(T);
        CHECK(cudaMemcpy(d_arr2, d_arr, tam, cudaMemcpyDeviceToDevice));
    }

    /* template to create and copy an array from host to device */
    template <class T>
    T* tod(T* h_arr, unsigned int n){
        T* arr = array<T>(n);
        h2d<T>(h_arr, arr, n);
        return arr;
    }

    /* template to create and copy an array from device to host */
    template <class T>
    T* toh(T* d_arr, unsigned int n){
        T* arr = new T[n];
        d2h<T>(d_arr, arr, n);
        return arr;
    }

    /*  template to fill array with a value*/
    template<class T>
    __global__ void k_fill(T* d_arr, T val, int n){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n){
            d_arr[i] = val;
        }
    }

    template<class T>
    void fill(T*& d_arr, T val, int n, unsigned int block_dim = 256){
        dim3 block(block_dim, 1, 1);
        dim3 grid(ceil(n / float(block.x)), 1, 1);
        // k_fill<T><<<ceil(float(n) / block_dim), block_dim>>>(d_arr, val, n);
        k_fill<T><<<grid, block>>>(d_arr, val, n);
    }

}

namespace sfu{
    void cv2sf(cv::Mat frameRGB, sf::Sprite& sprite, sf::Texture& texture);
}


namespace cvu{
    /* todo: change indices i,j to j,i*/
    template<class T>
    void T2mat(cv::Mat& mat, unsigned int height, unsigned int width, T* data, T range);
}





#endif
