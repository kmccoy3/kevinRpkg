

#ifndef EM_HEADER_GUARD
#define EM_HEADER_GUARD


#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]


bool is_positive_definite(const Eigen::MatrixXd &A);
double mvn_pdf(const Eigen::VectorXd &x,
               const Eigen::VectorXd &mu,
               const Eigen::MatrixXd &Sigma);
Rcpp::List EM(const Eigen::MatrixXd &X,
              const int num_groups,
              const int max_iter = 1000,
              const double eps = 1e-3,
              const int verbose = 0);

double expected_log_likelihood(const int n,
                               const int num_groups,
                               Eigen::VectorXd& tau,
                               Eigen::MatrixXd& mu,
                               std::vector<Eigen::MatrixXd>& Sigma,
                               const Eigen::MatrixXd& X,
                               Eigen::MatrixXd& T);



void expectation_step(const int n,
                      const int num_groups,
                      const Eigen::VectorXd& tau,
                      const Eigen::MatrixXd& mu,
                      const std::vector<Eigen::MatrixXd>& Sigma,
                      const Eigen::MatrixXd& X,
                      Eigen::MatrixXd& T);

void maximization_step(const int n,
                       const int p,
                       const int num_groups,
                       Eigen::VectorXd& tau,
                       Eigen::MatrixXd& mu,
                       std::vector<Eigen::MatrixXd>& Sigma,
                       const Eigen::MatrixXd& X,
                       const Eigen::MatrixXd& T);

#endif

