#include <iostream>
#include <vector>
#include <tuple>

#include <SFML/Graphics.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/videoio/videoio.hpp"


#include "gui.h"





int main(){
  
  sf::RenderWindow window(sf::VideoMode(win_width, win_height), "RenderWindow");
  tgui::Gui gui{window}; 
  
  sf::Texture text_curr, text_test;
  sf::Sprite spr_curr, spr_test;
  cv::Mat mat_curr, mat_test;
  sf::Event event;


  int all_val_add = 0;
  float all_val_prod = 0;
  
  inicializar_gui(std::ref(gui), std::ref(mat_curr), std::ref(text_curr), std::ref(spr_curr), std::ref(mat_test), std::ref(text_test), std::ref(spr_test));
  
  tgui::Slider::Ptr slider_all_add = gui.get<tgui::Slider>("SliderAddAll");
  tgui::Slider::Ptr slider_all_prod = gui.get<tgui::Slider>("SliderProdAll");
  
  while (window.isOpen())  {

    while (window.pollEvent(event))    {
      if (event.type == sf::Event::Closed)
        window.close();
      if (event.type == sf::Event::MouseButtonReleased){
        if(all_val_add != (int)slider_all_add->getValue()){
          f_addImage(gui, (int)slider_all_add->getValue());
          all_val_add = (int)slider_all_add->getValue();
          cout<<(int)slider_all_add->getValue()<<endl;
        }
        if(all_val_prod != (int)slider_all_prod->getValue()){
          f_prodImage(gui, slider_all_prod->getValue() / 25.0);
          all_val_prod = slider_all_prod->getValue();
          cout<<(int)slider_all_prod->getValue()<<endl;
        }
      }
      gui.handleEvent(event);
    }
    window.clear();
    gui.draw();
    window.draw(spr_curr);
    window.draw(spr_test);
    window.display();
  }
  return 0;


}