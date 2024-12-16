% Generate in MATLAB the data from the paper Estimating Context Effects in Small Samples 
% while Controlling for Covariates: An Optimally Regularized Bayesian Estimator 
% for Multilevel Latent Variable Models
% Authors:
% Valerii Dashuk (Medical School Hamburg),
% Martin Hecht (Helmut Schmidt University),
% Oliver Lüdtke (Leibniz Institute for Science and Mathematics Education, Centre for International Student Assessment),
% Alexander Robitzsch (Leibniz Institute for Science and Mathematics Education, Centre for International Student Assessment),
% Steffen Zitzmann (Medical School Hamburg)
%
% 540 cases with 5000 replications for each case

ICC_x = [0.05; 0.1; 0.3; 0.5];

J = [5; 10; 20; 30; 40];

n = [5; 15; 30];

beta_b = [0.2; 0.5; 0.6];

beta_w =[0.2; 0.5; 0.7];

rep = 5000;

c = 1 % counter for number of cases

for j1=1:length(ICC_x)
    for j2=1:length(J)
        for j3=1:length(n)
            for j4=1:length(beta_b)
                for j5=1:length(beta_w)

                    % Seed and generate data. Use k as seed.
                    rng(k, 'twister');

                    for i=1:rep
                        Data(i).data = ALCD_CV(j1, j2, j3, j4, j5); % function 
                    end

		    c = c + 1;

                end
            end
        end
    end
end

c = c - 1; % adjust the number of cases


% Core function that generates data by given parameters
function data = ALCD_CV(ICC_x, k, n, beta_b, beta_w)

% creates a structure with simulated data and its parameters for 2-level
% model with latent covariate, k groups, n elements in each group

% define the values for k groups, n elements in each group

ICC_y = 0.3; % for further references: tau_y2 = ICC_y; sigma_y2 = 1 - ICC_y

% coefficients to estimate

b0 = 0;
b1 = beta_w; % within coefficient b_w
b2 = beta_b; % between coefficient b_b

% control variable(s) coefficient(s)
gamma = 0.3;

kc = 1; % number of control variables, if change kc, also change gamma, m_C, ICC_C, cov_mat, cov_mat_b % CV code

% check for the possible value of ICC_x
% check for the possible value of ICC_x
if ICC_x < max(1 - (1 - ICC_y) / b1^2, 0)
    error('Too small value of ICC_x. The value should be larger than %f', max(1 - (1 - ICC_y) / b1^2, 0));
elseif ICC_x > min(ICC_y / b2^2, 1)
    error('Too large value of ICC_x. The value should be smaller than %f', min(ICC_y / b2^2, 1));
elseif k <= 0 || n <= 0
    error('<strong>k</strong> or <strong>n</strong> are negative. Consider natural numbers.');
elseif floor(k) ~= ceil(k) || floor(n) ~= ceil(n)
    error('<strong>k</strong> or <strong>n</strong> are non-integers. Consider using natural numbers.');
else

    kn = k*n; % total number of observations
    
    % adding control variables:

    m_C = 0; % mean vector of control variables

    ICC_C = 0.4; % ICCs of control variables

    % create the covariance matrices total, between and within parts of
    % matrix [x,C] of size (kc+1) x kn and check for eligibility of inputs:
    
    % covariance matrix should have the size (kc+1) x (kc+1)
    cov_mat = [1, 0.4; 0.4, 1];
    cov_mat_b = diag([ICC_x; ICC_C]) + [0, 0.1; 0.1, 0];

    cov_mat_w = cov_mat - cov_mat_b; % CV code

    % check that all covariance matrices are positive semidefinite

    if ~issymmetric(cov_mat) || ~issymmetric(cov_mat_b) || ~issymmetric(cov_mat_w) % CV code
         error('Covariance matrix is not symmetric.');
    end
    
    if ~all(eig(cov_mat) > -0.00001) || ~all(eig(cov_mat_b) > -0.00001) || ~all(eig(cov_mat_w) > -0.00001)
        error('Covariance matrix is not positive semidefinite.');
    end
    

    % Variances and covariances
    var_C1 = eye(kc) - diag(ICC_C); % within variance matrix of control variables % CV code
    var_C2 = diag(ICC_C); % between variance matrix of control variables % CV code

    m_x = 0; % mean
    var_x1 = 1 - ICC_x; % within variance sigma_x2
    var_x2 = ICC_x; % between variance tau_x2

    % extract covariance as vector from covariance matrix
    
    cov_xC = cov_mat(logical(tril(ones(size(cov_mat)), -1))); % CV code % overall covariances between all elements of x and C that go column by column, like cov_xC = [cov(x,C1); cov(x,C2); cov(x,C3); cov(C1,C2); cov(C1,C3); cov(C2,C3)];
    ICC_xC = cov_mat_b(logical(tril(ones(size(cov_mat_b)), -1))); % CV code % between part covariances of all elements of x and C that go column by column, like cov_xC = [cov(x,C1); cov(x,C2); cov(x,C3); cov(C1,C2); cov(C1,C3); cov(C2,C3)];
    
    var_e2 = ICC_y - b2^2 * ICC_x - (gamma.^2)' * ICC_C - prod(nchoosek([b2; gamma], 2)') * ICC_xC; % between residual variance
    var_e1 = (1 - ICC_y) - b1^2 * (1 - ICC_x) - (gamma.^2)' * (1 - ICC_C) - prod(nchoosek([b1; gamma], 2)')* (cov_xC - ICC_xC); % within residual variance

    if var_e1 <= 0 || var_e2 <= 0
        error('Values of ICC_x, ICC_C, and ICC_y do not coincide.');
    end
    
    % random data generation
    xC2 = mvnrnd1([m_x; m_C], cov_mat_b, k); 
    xC = mvnrnd1(zeros(kc+1,1), cov_mat_w, kn) + kron(xC2,ones(n,1)); 
    C2 = xC2(:,2:end); % between part regresor, k elements 
    C = xC(:,2:end); % overall control variable 
    x2 = xC2(:,1); % between part regresor, k elements 
    e2 = randn(k,1)*sqrt(var_e2); % between part residual, k elements 
    x = xC(:,1); % overall regressor, k*n elements 

    % simulate output y, with theoretical values:
    % within covariance of y and x sigma_yx = b1*var_x1 and 
    % between covariance of y and x tau_yx = b2*var_x2
    % within covariance of y and C sigma_yC = var_C1*gamma
    % between covariance of y and C tau_yC = var_C2*gamma
    y = randn(kn,1)*sqrt(var_e1) + (b0 + b1*(x - kron(x2,ones(n,1))) + b2*kron(x2,ones(n,1)) + kron(e2,ones(n,1))) + C*gamma; % output, k*n elements
    
    % group averages
    x2D = zeros(k,1);
    for i=1:k
        x2D(i) = mean(x(n*(i-1)+1:n*i));
    end
    
    % combine all in one structure
    data.k = k;
    data.n = n;
    data.kc = kc; 
    data.ICC_x = ICC_x;
    data.ICC_y = ICC_y;
    data.ICC_C = ICC_C; 
    data.b0 = b0;
    data.b_w = b1;
    data.b_b = b2;
    data.gamma = gamma; 
    data.kn = kn;
    data.m_x = m_x;
    data.var_x1 = var_x1;
    data.var_x2 = var_x2;
    data.var_e1 = var_e1;
    data.var_e2 = var_e2;
    data.cov_mat = cov_mat; 
    data.cov_mat_b = cov_mat_b; 
    data.x2 = x2;
    data.x = x;
    data.e2 = e2;
    data.m_C = m_C;
    data.var_C1 = var_C1; 
    data.var_C2 = var_C2; 
    data.C2 = C2; 
    data.C = C;
    data.y = y;
    data.x2D = x2D;
end
end