#include "gui.h"



uchar* d_data;
uchar* h_data;
uchar* d_temp;
uchar* h_temp;

int img_w = win_width * img_width;
int img_h = win_height * img_height;

dim3 block = dim3(16, 16, 1);
dim3 grid = dim3(ceil( img_h/ float(block.x)), ceil(img_w / float(block.y)));

cv::Mat* mat_test_p;
sf::Texture* text_test_p;
sf::Sprite* spr_test_p;
cv::Mat* mat_curr_p;
sf::Texture* text_curr_p;
sf::Sprite* spr_curr_p;

/*template matching*/
cv::Mat templ;
float* d_r;
float* d_g;
float* d_b;


/*convolutions*/
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


__global__ void conv(unsigned char* d_data, unsigned char* d_temp, float* d_kernel, int k, int p, int pos_k, int pos_p, int h, int w){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
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
        d_temp[j * 3 * w + i * 3] = sum_c1;
        d_temp[j * 3 * w + i * 3 + 1] = sum_c2;
        d_temp[j * 3 * w + i * 3 + 2] = sum_c3;
    }
}


__global__ void template_matching(uchar* d_data, uchar* d_temp, float* d_kernel1, float* d_kernel2, float* d_kernel3, int k, int p, int pos_k, int pos_p, int h, int w){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < w && j < h){
        float sum_c1 = 0;
        float sum_c2 = 0;
        float sum_c3 = 0;
        for (size_t r = 0; r < k; r++) {
            for (size_t c = 0; c < p; c++) {
                sum_c1 += fabs(d_data[(j - pos_k + r) * (3 * w) + ((i - pos_p + c) * 3)] - d_kernel1[r * p + c]);
                sum_c2 += fabs(d_data[(j - pos_k + r) * (3 * w) + ((i - pos_p + c) * 3) + 1] - d_kernel2[r * p + c]);
                sum_c3 += fabs(d_data[(j - pos_k + r) * (3 * w) + ((i - pos_p + c) * 3) + 2] - d_kernel3[r * p + c]);
            }
        }
        float val = sum_c1 + sum_c2 + sum_c3;
        val = val / (p * k * 3);
  
        d_temp[j * 3 * w + i * 3] = val;
        d_temp[j * 3 * w + i * 3 + 1] = val;
        d_temp[j * 3 * w + i * 3 + 2] = val;  
      
    }
  }
  

__global__ void addImage(uchar* d_data, uchar* d_temp, int val, unsigned int height, unsigned int width){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(j < height && i < width){
        int r, g, b;
        r = at3(d_data, j, i, 0, width, 3) + val;
        g = at3(d_data, j, i, 1, width, 3) + val;
        b = at3(d_data, j, i, 2, width, 3) + val;

        if(r > 255)
            r = 255;
        if(g > 255)
            g = 255;
        if(b > 255)
            b = 255;

        if(r < 0)
            r = 0;
        if(g < 0)
            g = 0;
        if(b < 0)
            b = 0;

        at3(d_temp, j, i, 0, width, 3) = r;
        at3(d_temp, j, i, 1, width, 3) = g;
        at3(d_temp, j, i, 2, width, 3) = b;
    }
}

__global__ void prodImage(uchar* d_data, uchar* d_temp, float val, unsigned int height, unsigned int width){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(j < height && i < width){
        int r, g, b;
        if(val > 0){
            r = at3(d_data, j, i, 0, width, 3) * val;
            g = at3(d_data, j, i, 1, width, 3) * val;
            b = at3(d_data, j, i, 2, width, 3) * val;
        }
        else{
            r = at3(d_data, j, i, 0, width, 3) / val;
            g = at3(d_data, j, i, 1, width, 3) / val;
            b = at3(d_data, j, i, 2, width, 3) / val;
        }

        if(r > 255)
            r = 255;
        if(g > 255)
            g = 255;
        if(b > 255)
            b = 255;

        if(r < 0)
            r = 0;
        if(g < 0)
            g = 0;
        if(b < 0)
            b = 0;

        at3(d_temp, j, i, 0, width, 3) = r;
        at3(d_temp, j, i, 1, width, 3) = g;
        at3(d_temp, j, i, 2, width, 3) = b;
    }
}

void f_applyFilter(tgui::Gui& gui, float* kernel, int k , int p, int i, int j){
    float* d_kernel = cudu::array<float>(k * p); 
  
    cudu::h2d<float>(kernel, d_kernel, k * p);
    conv<<<grid, block>>>(d_data, d_temp, d_kernel, k, p, i, j, img_h, img_w);
    cudaDeviceSynchronize();
    cudu::d2h<uchar>(d_temp, h_temp, img_h * img_w * 3);
    cudaFree(d_kernel);
    sfu::cv2sf(*mat_test_p, *spr_test_p, *text_test_p);
  
  }
  
  


void f_loadTemplate(tgui::Gui& gui){
    tgui::EditBox::Ptr editBox_template_name = gui.get<tgui::EditBox>("TemplateName");
    string nombre = std::string(editBox_template_name->getText()).c_str();

    templ = imread(nombre);
    resize(templ, templ, Size(templ.cols * img_width, templ.rows * img_height));

    int height, width;
    width = templ.cols;
    height = templ.rows;  

    float* h_r = new float[width * height];
    float* h_g = new float[width * height];
    float* h_b = new float[width * height];

    
    d_r = cudu::array<float>(height * width);
    d_g = cudu::array<float>(height * width);
    d_b = cudu::array<float>(height * width);

    unsigned char* data_templ = (unsigned char*)templ.data;

    for(int j = 0; j < height; j++){
        for(int i = 0; i < width; i++){
            h_r[j * width + i] = data_templ[j * width * 3 + i * 3];
            h_g[j * width + i] = data_templ[j * width * 3 + i * 3 + 1];
            h_b[j * width + i] = data_templ[j * width * 3 + i * 3 + 2];
        }
    }

    cudu::h2d<float>(h_r, d_r, width * height);
    cudu::h2d<float>(h_g, d_g, width * height);
    cudu::h2d<float>(h_b, d_b, width * height);
}


void f_applyTemplate(tgui::Gui& gui){
    template_matching<<<grid, block>>>(d_data, d_temp, d_r, d_g, d_b, templ.rows, templ.cols, templ.rows/2, templ.cols/2, img_h, img_w);
    devs();

    cudu::d2h<uchar>(d_temp, h_temp, img_h * img_w * 3);

    sfu::cv2sf(*mat_test_p, *spr_test_p, *text_test_p);

}


void f_addImage(tgui::Gui& gui, int val){
    addImage<<<grid, block>>>(d_data, d_temp, val, img_h, img_w);
    devs();
    cudu::d2h<uchar>(d_temp, h_temp, img_h * img_w * 3);

    // sfu::cv2sf(mat_curr, spr_curr, text_curr);
    sfu::cv2sf(*mat_test_p, *spr_test_p, *text_test_p);
}

void f_prodImage(tgui::Gui& gui, float val){
    prodImage<<<grid, block>>>(d_data, d_temp, val, img_h, img_w);
    devs();
    cudu::d2h<uchar>(d_temp, h_temp, img_h * img_w * 3);

    // sfu::cv2sf(mat_curr, spr_curr, text_curr);
    sfu::cv2sf(*mat_test_p, *spr_test_p, *text_test_p);
}

void f_apply(){
    std::copy(h_temp, h_temp + (img_h * img_w * 3), h_data);
    sfu::cv2sf(*mat_curr_p, *spr_curr_p, *text_curr_p);
    
}

void f_loadImage(tgui::Gui& gui, cv::Mat& mat_curr, sf::Texture& text_curr, sf::Sprite& spr_curr, cv::Mat& mat_test, sf::Texture& text_test, sf::Sprite& spr_test){


    tgui::EditBox::Ptr editBox_image_name = gui.get<tgui::EditBox>("ImageName");
    string nombre = std::string(editBox_image_name->getText()).c_str();

    mat_curr = imread(nombre);
    resize(mat_curr, mat_curr, Size(img_w, img_h));
    mat_test = mat_curr.clone();
    sfu::cv2sf(mat_curr, spr_curr, text_curr);
    sfu::cv2sf(mat_test, spr_test, text_test);


    
    h_temp = (uchar*)mat_test.data;
    h_data = (uchar*)mat_curr.data;
    cudu::h2d<uchar>(h_data, d_data, img_w * img_h * 3);
    

}


void inicializar_gui(tgui::Gui& gui, cv::Mat& mat_curr, sf::Texture& text_curr, sf::Sprite& spr_curr, cv::Mat& mat_test, sf::Texture& text_test, sf::Sprite& spr_test){
    tgui::Theme theme{"Black.txt"};
    tgui::Theme::setDefault(&theme);

    auto picture = tgui::Picture::create("fondo2.jpg");
    picture->setSize({"100%", "100%"});
    gui.add(picture);


    tgui::Button::Ptr load_button = tgui::Button::create();
    tgui::Button::Ptr apply_button = tgui::Button::create();
    tgui::EditBox::Ptr editBox_image_name = tgui::EditBox::create();
    tgui::Slider::Ptr slider_add_all = tgui::Slider::create();
    tgui::Slider::Ptr slider_prod_all = tgui::Slider::create();
    tgui::Button::Ptr apply_template = tgui::Button::create();
    tgui::Button::Ptr load_template = tgui::Button::create();
    tgui::EditBox::Ptr editBox_template_name = tgui::EditBox::create();
    tgui::Button::Ptr apply_perfilado = tgui::Button::create();
    tgui::Button::Ptr apply_sobelx = tgui::Button::create();
    tgui::Button::Ptr apply_sobely = tgui::Button::create();
    tgui::Button::Ptr apply_filtrogaussianox = tgui::Button::create();
    tgui::Button::Ptr apply_filtrogaussianoy = tgui::Button::create();


    gui.add(load_button, "LoadImage");
    gui.add(apply_button, "Apply");
    gui.add(editBox_image_name, "ImageName");
    gui.add(slider_add_all, "SliderAddAll");
    gui.add(slider_prod_all, "SliderProdAll");
    gui.add(apply_template, "ApplyTemplate");
    gui.add(load_template, "LoadTemplate");
    gui.add(editBox_template_name, "TemplateName");
    gui.add(apply_perfilado, "Perfilado");
    gui.add(apply_sobelx, "SobelX");
    gui.add(apply_sobely, "SobelY");
    gui.add(apply_filtrogaussianox, "FiltroGaussianoX");
    gui.add(apply_filtrogaussianoy, "FiltroGaussianoY");



    load_button->setText("Load image");
    load_button->setPosition("25%", "10%");

    apply_button->setText("Apply changes");
    apply_button->setPosition("50%", "35%");

    editBox_image_name->setText("a.png");
    editBox_image_name->setPosition("5%", "10%");

    slider_add_all->setPosition("5%", "20%");
    slider_add_all->setMaximum(255.0);
    slider_add_all->setMinimum(-255.0);
    slider_add_all->setValue(122);
    

    slider_prod_all->setPosition("5%", "30%");
    slider_prod_all->setMaximum(255.0);
    slider_prod_all->setMinimum(-255.0);
    slider_prod_all->setValue(1);

    apply_template->setText("Apply template");
    apply_template->setPosition("5%", "55%");

    load_template->setText("Load template");
    load_template->setPosition("25%", "45%");

    editBox_template_name->setText("b.png");
    editBox_template_name->setPosition("5%", "45%");

    apply_perfilado->setText("Pefilado");
    apply_perfilado->setPosition("10%", "65%");

    apply_sobelx->setText("Sobel x");
    apply_sobelx->setPosition("20%", "65%");

    apply_sobely->setText("Sobel y");
    apply_sobely->setPosition("30%", "65%");

    apply_filtrogaussianox->setText("Filtro Gaussiano x");
    apply_filtrogaussianox->setPosition("40%", "65%");

    apply_filtrogaussianoy->setText("Filtro Gaussiano y");
    apply_filtrogaussianoy->setPosition("50%", "65%");


    /* setting initial images */
    mat_curr = imread("noimage.png");
    resize(mat_curr, mat_curr, Size(img_w, img_h));
    mat_test = mat_curr.clone();
    sfu::cv2sf(mat_curr, spr_curr, text_curr);
    sfu::cv2sf(mat_test, spr_test, text_test);



    spr_curr.setPosition(win_width * 0.46, win_height * 0.1);
    spr_test.setPosition(win_width * 0.75, win_height * 0.1);




    
    /* botones funciones */
    load_button->connect("pressed", f_loadImage, std::ref(gui), std::ref(mat_curr), std::ref(text_curr), std::ref(spr_curr), std::ref(mat_test), std::ref(text_test), std::ref(spr_test));
    apply_button->connect("pressed", f_apply);
    load_template->connect("pressed", f_loadTemplate, std::ref(gui));
    apply_template->connect("pressed", f_applyTemplate, std::ref(gui));
    apply_perfilado->connect("pressed", f_applyFilter, std::ref(gui), &kernel_perfilado[0], 3, 3, 1, 1);
    apply_sobelx->connect("pressed", f_applyFilter, std::ref(gui), &kernel_Sobel_X[0], 3, 3, 1, 1);
    apply_sobely->connect("pressed", f_applyFilter, std::ref(gui), &kernel_Sobel_Y[0], 3, 3, 1, 1);
    apply_filtrogaussianox->connect("pressed", f_applyFilter, std::ref(gui), &kernel_filtro_gaussiano_l[0], 1, 7, 3, 0);
    apply_filtrogaussianoy->connect("pressed", f_applyFilter, std::ref(gui), &kernel_filtro_gaussiano_l[0], 7, 1, 3, 0);
    

    mat_test_p = &mat_test;
    spr_test_p = &spr_test;
    text_test_p = &text_test;
    mat_curr_p = &mat_curr;
    spr_curr_p = &spr_curr;
    text_curr_p = &text_curr;    

    d_data = cudu::array<uchar>(mat_curr.cols * mat_curr.rows * 3);
    d_temp = cudu::array<uchar>(mat_curr.cols * mat_curr.rows * 3);
    h_data = (uchar*)mat_curr.data;
    h_temp = (uchar*)mat_test.data;
}

