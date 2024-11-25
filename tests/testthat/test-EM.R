
library(testthat)
library(MASS)

## test of placeholder function hello()



X = c(0, 0)
mu = c(0, 0)
Sigma = matrix(c(1, 0, 0, 1), nrow=2)


test_that("mvn_pdf works", {
    expect_equal(mvn_pdf(X, mu, Sigma), 0.1591549, tolerance = 1e-3)
})


################################################################################

N <- 1000
tau <- 0.7
mu_1 <- c(0.25, 0.25)
mu_2 <- c(0.75, 0.75)
sigma_1 <- 0.1*matrix(c(1, 0.4, 0.4, 1), nrow=2)
sigma_2 <- 0.01*matrix(c(1, -0.4, -0.4, 1), nrow=2)

data_1 <- MASS::mvrnorm(n=N*tau, mu=mu_1, Sigma=sigma_1)
data_2 <- MASS::mvrnorm(n=N*(1-tau), mu=mu_2, Sigma=sigma_2)

data <- as.data.frame(rbind(data_1, data_2))
colnames(data) <- c("X", "Y")
data$cluster <- as.factor(c(rep(1, N*tau), rep(2, N*(1-tau))))

res <- EM(as.matrix(data[,-3]), 2, max_iter=1000, eps=1e-6, verbose=0)

is_reverse <- res$tau[1] < 0.5

if (is_reverse) {
    res$tau <- 1 - res$tau
    res$mu <- res$mu[2:1, ]
    res$sigma <- res$sigma[2:1]
}


test_that("Get expected result from EM", {
    expect_equal(res$tau[1], tau, tolerance = 0.1)
    expect_equal(res$mu[1,], mu_1, tolerance = 0.1)
    expect_equal(res$mu[2,], mu_2, tolerance = 0.1)
    expect_equal(res$Sigma[[1]], sigma_1, tolerance = 0.1)
    expect_equal(res$Sigma[[2]], sigma_2, tolerance = 0.1)
})















