#ifndef GUI_H
#define GUI_H

#include <iostream>
#include <TGUI/TGUI.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/videoio/videoio.hpp"


#include "utils.h"


#define win_width 1600
#define win_height 800
#define img_width 0.22
#define img_height 0.25


using namespace cv;
using namespace std;

void f_applyFilter(tgui::Gui& gui, float* kernel, int k , int p, int i, int j);

void f_addImage(tgui::Gui& gui, int val);

void f_prodImage(tgui::Gui& gui, float val);

void f_loadTemplate(tgui::Gui& gui);

void f_applyTemplate(tgui::Gui& gui);

void f_loadImage(tgui::Gui& gui, cv::Mat& mat_curr, sf::Texture& text_curr, sf::Sprite& spr_curr, cv::Mat& mat_test, sf::Texture& text_test, sf::Sprite& spr_test);

void inicializar_gui(tgui::Gui& gui, cv::Mat& mat_curr, sf::Texture& text_curr, sf::Sprite& spr_curr, cv::Mat& mat_test, sf::Texture& text_test, sf::Sprite& spr_test);


    

#endif