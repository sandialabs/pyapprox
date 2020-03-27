#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <dolfin/function/Expression.h>
#include <iostream>

# define M_PI           3.14159265358979323846  // pi

class NobileDiffusivityExpression : public dolfin::Expression{
public:
  double clen_,L_,Lp_,mean_field_;
  int nvars_;
  std::vector<double> eigenvalues_;

  Eigen::VectorXd sample_;
  
public:
  NobileDiffusivityExpression() : dolfin::Expression(),clen_(0),L_(0),Lp_(0),mean_field_(0.5),nvars_(0)
  {}
  
  void initialize_kle(int nvars, double clen){
    nvars_=nvars; clen_=clen;

    double domain_len=1;
    Lp_ = std::max(domain_len,2*clen_);
    L_ =  clen_/Lp_;
    eigenvalues_.resize(nvars_-1);
    for (int ii=0;ii<nvars_-1;ii++){
      eigenvalues_[ii] = std::sqrt(std::sqrt(M_PI)*L_)*
	//std::exp(-std::pow(((ii+2)/2*M_PI*L_),2)/8.0);
	std::exp(-std::pow(((ii+2)/2*M_PI*L_),2)/4.0);
    }
  }

  void set_mean_field(double mean_field){
    mean_field_=mean_field;
  }

  void set_random_sample(Eigen::Ref<const Eigen::VectorXd> sample){
    sample_=sample;
  }
  
  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const{
    values[0]=1.0+sample_[0]*std::sqrt(std::sqrt(M_PI)*L_/2.0);
    for (int ii=0;ii<nvars_-1;ii+=2){
      values[0]+=eigenvalues_[ii]*std::sin((ii+2)/2*M_PI*x[0]/Lp_)*sample_[ii+1];
    }
    for (int ii=1;ii<nvars_-1;ii+=2){
      values[0]+=eigenvalues_[ii]*std::cos((ii+2)/2*M_PI*x[0]/Lp_)*sample_[ii+1];
    }
    values[0]=std::exp(values[0])+mean_field_;
    //std::printf("v=%1.2f\n",values[0]);
    //std::printf("m=%1.2f\n",mean_field_);
  }
};

PYBIND11_MODULE(SIGNATURE, m) {
  pybind11::class_<NobileDiffusivityExpression, std::shared_ptr<NobileDiffusivityExpression>, dolfin::Expression>
    (m, "NobileDiffusivityExpression")
    .def(pybind11::init<>())
    //.def_readwrite("sample_", &NobileDiffusivityExpression::sample_)
    .def("initialize_kle",&NobileDiffusivityExpression::initialize_kle,
	 "init kle")
    .def("set_random_sample",&NobileDiffusivityExpression::set_random_sample,
	 "set sample")
    .def("set_mean_field",&NobileDiffusivityExpression::set_mean_field,
	 "set mean field");
  ;
}
