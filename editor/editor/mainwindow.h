#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPixmap>
#include <QImage>
#include <QDebug>
#include <QPushButton>
#include <QComboBox>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <math.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <QCoreApplication>

//#include "funciones.h"

using namespace std;
using namespace cv;





namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();


private slots:
    void handleButton();
    void show_image();
    void load_image();
    void operacion();
private:
    Ui::MainWindow *ui;
    string path = "/home/texs/MEGA/UNSA/9no semestre/grafica/topicos_computacion_grafica/editor/imagenes/";
    QPushButton *m_button;
    QPushButton *b_aplicar;
    QComboBox* c_operaciones;
    Mat img;
    Mat hist;

};

#endif // MAINWINDOW_H
