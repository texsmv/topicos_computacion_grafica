#include "mainwindow.h"
#include "ui_mainwindow.h"


Mat histograma_m(Mat src){
    vector<Mat> bgr_planes;
    split( src, bgr_planes );
    int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange = { range };
    bool uniform = true, accumulate = false;
    Mat b_hist, g_hist, r_hist;
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ),
              Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
              Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ),
              Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
              Scalar( 0, 255, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ),
              Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
              Scalar( 0, 0, 255), 2, 8, 0  );
    }
    return histImage;
}


Mat suma_m(Mat img_i, float k){
  Mat img = img_i.clone();
  unsigned char *input = (unsigned char*)(img.data);
  for(int i = 0;i < img.rows;i++){
      for(int j = 0;j < img.cols;j++){
          // unsigned char b = input[img.step * j + i ] ;
          // unsigned char g = input[img.step * j + i + 1];
          // unsigned char r = input[img.step * j + i + 2];
          if(input[img.cols * 3 * i + j * 3 ] + k > 255){
              input[img.cols * 3 * i + j * 3 ] = 255;
          }else
              input[img.cols * 3 * i + j * 3 ] += k;
          if(input[img.cols * 3 * i + j * 3 + 1] + k > 255){
              input[img.cols * 3 * i + j * 3 + 1] = 255;
          }else
            input[img.cols * 3 * i + j * 3 + 1] += k;
          if(input[img.cols * 3 * i + j * 3 + 2] + k > 255){
              input[img.cols * 3 * i + j * 3 + 2] = 255;
          }else
              input[img.cols * 3 * i + j * 3 + 2] += k;
      }
  }
  return img;
}

Mat resta_m(Mat img_i, float k){
  Mat img = img_i.clone();
  unsigned char *input = (unsigned char*)(img.data);
  for(int i = 0;i < img.rows;i++){
      for(int j = 0;j < img.cols;j++){
          // unsigned char b = input[img.step * j + i ] ;
          // unsigned char g = input[img.step * j + i + 1];
          // unsigned char r = input[img.step * j + i + 2];
          if(input[img.cols * 3 * i + j * 3 ] - k < 0){
              input[img.cols * 3 * i + j * 3 ] = 0;
          }else
              input[img.cols * 3 * i + j * 3 ] -= k;
          if(input[img.cols * 3 * i + j * 3 + 1] - k < 0){
              input[img.cols * 3 * i + j * 3 + 1] = 0;
          }else
            input[img.cols * 3 * i + j * 3 + 1] -= k;
          if(input[img.cols * 3 * i + j * 3 + 2] - k < 0){
              input[img.cols * 3 * i + j * 3 + 2] = 0;
          }else
              input[img.cols * 3 * i + j * 3 + 2] -= k;
      }
  }
  return img;
}



Mat multiplicacion_m(Mat img_i, float k){
  Mat img = img_i.clone();
  unsigned char *input = (unsigned char*)(img.data);
  for(int i = 0;i < img.rows;i++){
      for(int j = 0;j < img.cols;j++){
          // unsigned char b = input[img.step * j + i ] ;
          // unsigned char g = input[img.step * j + i + 1];
          // unsigned char r = input[img.step * j + i + 2];
          if(input[img.cols * 3 * i + j * 3 ] * k > 255){
              input[img.cols * 3 * i + j * 3 ] = 255;
          }else
              input[img.cols * 3 * i + j * 3 ] *= k;
          if(input[img.cols * 3 * i + j * 3 + 1] * k > 255){
              input[img.cols * 3 * i + j * 3 + 1] = 255;
          }else
            input[img.cols * 3 * i + j * 3 + 1] *= k;
          if(input[img.cols * 3 * i + j * 3 + 2] * k > 255){
              input[img.cols * 3 * i + j * 3 + 2] = 255;
          }else
              input[img.cols * 3 * i + j * 3 + 2] *= k;
      }
  }
  return img;
}

Mat division_m(Mat img_i, float k){
  Mat img = img_i.clone();
  unsigned char *input = (unsigned char*)(img.data);
  for(int i = 0;i < img.rows;i++){
      for(int j = 0;j < img.cols;j++){
          // unsigned char b = input[img.step * j + i ] ;
          // unsigned char g = input[img.step * j + i + 1];
          // unsigned char r = input[img.step * j + i + 2];
          if(input[img.cols * 3 * i + j * 3 ] / k < 0){
              input[img.cols * 3 * i + j * 3 ] = 0;
          }else
              input[img.cols * 3 * i + j * 3 ] /= k;
          if(input[img.cols * 3 * i + j * 3 + 1] / k < 0){
              input[img.cols * 3 * i + j * 3 + 1] = 0;
          }else
            input[img.cols * 3 * i + j * 3 + 1] /= k;
          if(input[img.cols * 3 * i + j * 3 + 2] / k < 0){
              input[img.cols * 3 * i + j * 3 + 2] = 0;
          }else
              input[img.cols * 3 * i + j * 3 + 2] /= k;
      }
  }
  return img;
}


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    m_button = (ui->load);
    b_aplicar = ui->aplicar;
    c_operaciones = ui->operacion;

    // Connect button signal to appropriate slot
    connect(m_button, SIGNAL (released()), this, SLOT (load_image()));
    connect(b_aplicar, SIGNAL (released()), this, SLOT (operacion()));

    load_image();
}

MainWindow::~MainWindow()
{
    delete ui;


}

void MainWindow::handleButton()
{
   // change the text
   show_image();
}


void MainWindow::operacion(){
    float k = atof(ui->valor_op->text().toStdString().c_str());
    if((c_operaciones->currentText()).toStdString() == "Suma"){
        img = suma_m(img, k);
    }
    else if((c_operaciones->currentText()).toStdString() == "Resta"){
        img = resta_m(img, k);
    }
    else if((c_operaciones->currentText()).toStdString() == "Multiplicacion"){
        img = multiplicacion_m(img, k);
    }
    else if((c_operaciones->currentText()).toStdString() == "Division"){
        img = division_m(img, k);
    }

    show_image();
}

void MainWindow::show_image(){

    hist = histograma_m(img);
    cv::resize(hist,hist,Size(200,200));
    cvtColor(hist, hist, CV_BGR2RGB);

    ui->imagen->setPixmap(QPixmap::fromImage(QImage((unsigned char*) img.data, img.cols, img.rows, QImage::Format_RGB888)));
//    ui->label_image1->setPixmap(QPixmap::fromImage(myImage));
    ui->imagen->show();

    ui->histograma->setPixmap(QPixmap::fromImage(QImage((unsigned char*) hist.data, hist.cols, hist.rows, QImage::Format_RGB888)));
//    ui->label_image1->setPixmap(QPixmap::fromImage(myImage));
    ui->histograma->show();
}


void MainWindow::load_image(){
    QImage myImage;
    img  = cv::imread(path + ui->nombre_img->text().toStdString());
//    cv::resize(mat, mat, cv::Size(), 0.25, 0.25);
    cvtColor(img, img, CV_BGR2RGB);


    show_image();



}
