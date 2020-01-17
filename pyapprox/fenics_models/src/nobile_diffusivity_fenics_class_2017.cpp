class NobileDiffusivityExpression : public Expression{
public:
  NobileDiffusivityExpression() : Expression(),nvars_(0),clen_(0),L_(0),Lp_(0)
  {}
  
  void initialize_kle(int nvars, double clen){
    nvars_=nvars; clen_=clen;

    double domain_len=1;
    Lp_ = std::max(domain_len,2*clen_);
    L_ =  clen_/Lp_;
    eigenvalues_.resize(nvars_-1);
    for (int ii=0;ii<nvars_-1;ii++){
      eigenvalues_[ii] = std::sqrt(std::sqrt(pi)*L_)*
	//std::exp(-std::pow(((ii+2)/2*pi*L_),2)/8.0);
      std::exp(-std::pow(((ii+2)/2*pi*L_),2)/4.0);
      //std::printf("%1.16e\n",eigenvalues_[ii]);
    }
    sample_.resize(nvars_);
  }

  void set_random_sample(double z, int ii){
   sample_[ii]=z;
  }
  
  void eval(Array<double>& values, const Array<double>& x) const{
    values[0]=1.0+sample_[0]*std::sqrt(std::sqrt(pi)*L_/2.0);
    for (int ii=0;ii<nvars_-1;ii+=2){
      values[0]+=eigenvalues_[ii]*std::sin((ii+2)/2*pi*x[0]/Lp_)*sample_[ii+1];
      //std::cout<<eigenvalues_[ii]*std::sin((ii+2)/2*pi*x[0]/Lp_)<<std::endl;
    }
    for (int ii=1;ii<nvars_-1;ii+=2){
      values[0]+=eigenvalues_[ii]*std::cos((ii+2)/2*pi*x[0]/Lp_)*sample_[ii+1];
      //std::cout<<eigenvalues_[ii]*std::cos((ii+2)/2*pi*x[0]/Lp_)<<std::endl;
    }
    values[0]=std::exp(values[0])+0.5;
    //if (std::abs(x[1])<1e-12) std::cout<<values[0]<<","<<x[0]<<std::endl;
  }
public:
  double clen_,L_,Lp_;
  int nvars_;
  std::vector<double> eigenvalues_;
  std::vector<double> sample_;
  };
