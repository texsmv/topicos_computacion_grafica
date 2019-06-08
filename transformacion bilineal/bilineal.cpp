#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <tuple>

using namespace Eigen;
using namespace std;

void apply_bilineal(float* coeff_x, float* coeff_y, int x_i, int y_i, int& x_o, int& y_o){
  x_o = coeff_x[0] * x_i + coeff_x[1] * y_i + coeff_x[2] * (x_i * y_i) + coeff_x[3];
  y_o = coeff_y[0] * x_i + coeff_y[1] * y_i + coeff_y[2] * (x_i * y_i) + coeff_y[3];
}


void get_mat_bilineal(vector<pair<float, float> > input_points, vector<pair<float, float> > output_points, vector<float>& coeff_x, vector<float>& coeff_y){
  int n_points = 4;
  MatrixXf A(n_points, n_points);
  VectorXf b_i(n_points);
  VectorXf b_j(n_points);

  for (size_t i = 0; i < n_points; i++) {
    // A.row(i) << input_points[i].first, input_points[i].second, input_points[i].first * input_points[i].second, 1;
    A(i, 0) = input_points[i].first;
    A(i, 1) = input_points[i].second;
    A(i, 2) = input_points[i].first * input_points[i].second;
    A(i, 3) = 1;
    b_i(i) = output_points[i].first;
    b_j(i) = output_points[i].second;
  }

  // cout << "Here is the matrix A:\n" << A << endl;
  // cout << "Here is the vector b:\n" << b_i << endl;

  VectorXf x = A.colPivHouseholderQr().solve(b_i);
  VectorXf y = A.colPivHouseholderQr().solve(b_j);
  for (size_t i = 0; i < 4; i++) {
    coeff_x.push_back(x(i));
    coeff_y.push_back(y(i));
  }
  // cout<<"Solucion x: "<<endl<<x<<endl;
  //
  // cout<<"Solucion y: "<<endl<<y<<endl;

}
