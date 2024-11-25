

#include <RcppEigen.h> // RcppEigen package, includes Rcpp.h

#include "EM.h"

// [[Rcpp::depends(RcppEigen)]]

bool is_positive_definite(const Eigen::MatrixXd &A)
{
    const Eigen::LLT<Eigen::MatrixXd> lltOfA(A); // compute the Cholesky decomposition of A
    return (A.isApprox(A.transpose())) && (lltOfA.info() == Eigen::Success);
}

// [[Rcpp::export]]
double mvn_pdf(const Eigen::VectorXd &x,
               const Eigen::VectorXd &mu,
               const Eigen::MatrixXd &Sigma)
{
    
    const int d = x.size(); // Dimension of the data
    
    // Ensure Sigma is positive definite (necessary for the multivariate normal)
    if (!is_positive_definite(Sigma))
    {
        throw(Rcpp::exception("Sigma matrix is not symmetric and positive definite!"));
        return -1.0;
    }
    
    // Compute the determinant and inverse of the covariance matrix
    const double detSigma = Sigma.determinant();
    const Eigen::MatrixXd Sigma_inv = Sigma.inverse();
    
    // Compute the Mahalanobis distance: (x - mu)^T * Sigma_inv * (x - mu)
    const Eigen::VectorXd diff = x - mu;
    const double mahalanobisDist = diff.transpose() * Sigma_inv * diff;
    
    // Compute the density using the formula
    double density = 1.0 / (std::sqrt(std::pow(2 * M_PI, d) * detSigma)) *
        std::exp(-0.5 * mahalanobisDist);
    
    return density;
}

double expected_log_likelihood(const int n,
                               const int num_groups,
                               Eigen::VectorXd &tau,
                               Eigen::MatrixXd &mu,
                               std::vector<Eigen::MatrixXd> &Sigma,
                               const Eigen::MatrixXd &X,
                               Eigen::MatrixXd &T)
{
    
    double Q_new = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < num_groups; ++j)
        {
            Q_new += T(j, i) * log(tau[j] * mvn_pdf(X.row(i), mu.row(j), Sigma[j]));
        }
    }
    return Q_new;
}

void expectation_step(const int n,
                      const int num_groups,
                      const Eigen::VectorXd &tau,
                      const Eigen::MatrixXd &mu,
                      const std::vector<Eigen::MatrixXd> &Sigma,
                      const Eigen::MatrixXd &X,
                      Eigen::MatrixXd &T)
{
    
    for (size_t i = 0; i < n; ++i)
    {
        
        const double den = tau[0] * mvn_pdf(X.row(i), mu.row(0), Sigma[0]) +
            tau[1] * mvn_pdf(X.row(i), mu.row(1), Sigma[1]);
        
        for (size_t j = 0; j < num_groups; ++j)
        {
            
            const double num = tau[j] * mvn_pdf(X.row(i), mu.row(j), Sigma[j]);
            
            T(j, i) = num / den;
        }
    }
}

void maximization_step(const int n,
                       const int p,
                       const int num_groups,
                       Eigen::VectorXd &tau,
                       Eigen::MatrixXd &mu,
                       std::vector<Eigen::MatrixXd> &Sigma,
                       const Eigen::MatrixXd &X,
                       const Eigen::MatrixXd &T)
{
    
    for (size_t j = 0; j < num_groups; ++j)
    {
        
        const double sum_T = T.row(j).sum();
        
        // Update tau
        tau[j] = sum_T / n;
        
        // Update mu
        mu.row(j) = T.row(j) * X / sum_T;
        
        // Update Sigma
        const Eigen::MatrixXd diff = X.rowwise() - mu.row(j);
        Eigen::MatrixXd Sigma_j = Eigen::MatrixXd::Zero(p, p);
        
        for (size_t i = 0; i < n; ++i)
        {
            Sigma_j += T(j, i) * diff.row(i).transpose() * diff.row(i);
        }
        
        Sigma[j] = Sigma_j / sum_T;
    }
}

// [[Rcpp::export]]
Rcpp::List EM(const Eigen::MatrixXd &X,
              const int num_groups,
              const int max_iter,
              const double eps,
              const int verbose)
{
    
    const int n = X.rows(); // Number of observations
    const int p = X.cols(); // Dimension of the data
    
    // Initialize the parameters
    Eigen::VectorXd tau(num_groups);
    Eigen::MatrixXd mu = Eigen::MatrixXd::Random(num_groups, p); // Random initialization in [-1, 1]
    std::vector<Eigen::MatrixXd> Sigma(num_groups);
    Eigen::MatrixXd T(num_groups, n);
    
    // Initial estimates
    tau << 0.5, 0.5;
    Sigma[0] = Eigen::MatrixXd::Identity(p, p);
    Sigma[1] = Eigen::MatrixXd::Identity(p, p);
    
    // Current / old value of Q
    double Q = 0.0;
    
    // Main iteration loop
    for (int num_iter = 0; num_iter < max_iter; ++num_iter)
    {
        
        // Print iteration number
        if (verbose && (num_iter % 100 == 0))
            Rcpp::Rcout << "Iteration: " << num_iter << "\n";
        
        // Expectation Step (E Step)
        expectation_step(n, num_groups, tau, mu, Sigma, X, T);
        
        // Compute the Q function
        double Q_new = expected_log_likelihood(n, num_groups, tau, mu, Sigma, X, T);
        
        // M Step
        maximization_step(n, p, num_groups, tau, mu, Sigma, X, T);
        
        // Check for convergence
        if (abs(Q_new - Q) < eps)
        {
            if (verbose)
                Rcpp::Rcout << "Converged after " << num_iter << " iterations\n";
            break;
        }
        
        // Update Q
        Q = Q_new;
    }
    
    const Rcpp::List result = Rcpp::List::create(Rcpp::Named("tau") = tau,
                                                 Rcpp::Named("mu") = mu,
                                                 Rcpp::Named("Sigma") = Sigma);
    
    return result;
}
