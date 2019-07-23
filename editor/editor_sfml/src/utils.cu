#include "utils.h"





    void sfu::cv2sf(cv::Mat frameRGB, sf::Sprite& sprite, sf::Texture& texture){
        
        sf::Image image;

        cv::cvtColor(frameRGB, frameRGB, cv::COLOR_BGR2RGBA);
        image.create(frameRGB.cols, frameRGB.rows, frameRGB.ptr());
        texture.loadFromImage(image);
        sprite.setTexture(texture);
    }


    /* todo: change indices i,j to j,i*/
    template<class T>
    void cvu::T2mat(cv::Mat& mat, unsigned int height, unsigned int width, T* data, T range){
        T min = std::numeric_limits<T>::max(), max = -std::numeric_limits<T>::max();
        cv::Vec3b **ptrText;
        int d;
        T temp;
        ptrText = new cv::Vec3b*[height];
        for(int i = 0; i < height; ++i) {
            ptrText[i] = mat.ptr<cv::Vec3b>(i);
            for(int j = 0; j < width; ++j) {
                // d = uchar((at2(data, i, j, width) / range) * 255);
                temp = at2(data, i, j, width);
                if(temp > max){
                    max = temp;
                }
                if(temp < min){
                    min = temp;
                }
                d = uchar( temp * 255.0 / range);
                
                ptrText[i][j] = cv::Vec3b(d, d, d);
            }
        }
        // std::cout<<"min:  "<<min<<" max:  "<<max<<std::endl;
    }
