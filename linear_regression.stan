// Linear Regression
data {
  int<lower=0> N;  // Sample size
  int<lower=0> P;  // Number of predictors (intercept + covariates)
  vector[N] Y;     // Variate (response variable)
  matrix[N, P] X;  // Design matrix including a column (1s for intercept) and covariates
}

parameters {
  vector[P] b;                    // Intercept and Slopes
  real<lower=0, upper=20> sigma; // SD of normal distribution
}

transformed parameters {
  vector[N] mu; // Mean of normal distribution
  mu = X * b;   // Linear predictor
}

model {
  // Priors
  b ~ normal(0, 100);     // Non-informative prior
  sigma ~ uniform(0, 20); // Non-informative prior
  
  // Likelihood
  Y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] Y_sim;    // Simulated data from posterior for prediction check
  vector[N] log_lik;  // Log likelihood for model comparison using WAIC and PSIS-LOOCV
  
  for (i in 1:N)
    Y_sim[i] = normal_rng(mu[i], sigma);
  
  for (i in 1:N)
    log_lik[i] = normal_lpdf(Y[i] | mu[i], sigma);
}
