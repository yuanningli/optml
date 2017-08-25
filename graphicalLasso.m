function B = graphicalLasso(S,rho,thres,maxIter)
% solve the sparse estimation of precision matrix using graphical lasso
%   B = argmax logdet(Theta) - Tr(S * Theta) - rho*||Theta||_1
% Inputs:
%     S - p-by-p sample covariance
%     rho - penalization parameter
%     thres - threshold for convergence
%     maxIter - max number of iterations
% Output:
%     B - p-by-p precision matrix
%
% Reference: Friedman et al. 2007
% 2017 @ Yuanning Li, ynli@cmu.edu

p = size(S,1);
% initialize
W = S + rho*eye(p);
LL(1) = logdet(W) - trace(S/W) - rho*sum(sum(abs(inv(W))));

counter = 1;
while counter < maxIter
    for i = 1:p
        % select the block
        idx = setdiff(1:p,i);
        W11 = W(idx,idx);
        s12 = S(idx,i);
        % solve the lasso
        [V,D] = eig(W11);
        A = real(V * sqrt(D) * V');
        b = real(V / sqrt(D) * V') * s12;
        beta = lassoFISTA(A,b,rho,0.1,thres,maxIter);
        W(idx,i) = W11*beta;
    end
    counter = counter + 1;
    LL(counter) = logdet(W) - trace(S/W) - rho*sum(sum(abs(inv(W))));
    if abs(LL(counter) - LL(counter -1)) < thres
        break
    end
end
B = inv(W);


function beta = lassoFISTA(X,y,lambda,b,thres,maxIter)
% solving lasso using accelerated proximal gradient descent (FISTA)
% Inputs: 
%   X - n-by-p data
%   y - n-by-1 dependent variable
%   lambda - regularization
%   b - backtracking parameter
%   thres - converging threshold
%   maxIter - max number of iterations
% Outputs:
%   beta - p-by-1 weights
%
% 2017 @ Yuanning Li, ynli@cmu.edu

beta = zeros(size(X,2),1);
fAcc = zeros(maxIter,1);
beta1 = beta; % beta(k-1)
beta2 = beta; % beta(k-2)
k = 1;
while k < maxIter
    v = beta1 + (k-2)/(k+1)*(beta1 - beta2);
    t = backtrack(y,X,beta,lambda,b);
    beta = softThres(v - t * grad(y,X,beta),t,lambda);
    fAcc(k) = costFunc(y,X,beta,lambda);
    beta2 = beta1;
    beta1 = beta;
    k = k + 1;
    if k > 2 && abs(fAcc(k) - fAcc(k-1)) < thres
        break
    end
end

function g = grad(y,X,beta)
% compute gradiant of ||y-X*beta||^2
g = - 2 * X' * (y - X * beta);

function beta = softThres(x,t,lambda)
% soft thresholding with t and lambda
% Input:
%   x - current input vector (p x 1)
%   t - stepsize
%   lambda - parameter
% Output: 
%   beta - soft thresholded beta (p x 1)

beta = zeros(size(x));
for i = 1 : length(beta)
    if x(i) >= lambda * t
        beta(i) = x(i) - lambda * t;
    elseif x(i) <= -lambda * t
        beta(i) = x(i) + lambda * t;
    end
end

function h = costFunc(y,X,beta,lambda)
% evaluate lasso cost function 1/2*||y-X*beta||^2 + lambda *|beta|_1
h = 1/2*norm(y-X*beta,2).^2 + lambda*sum(abs(beta));

function t = backtrack(y,X,beta,lambda,b)
% backtracking line search for stepsize
% Input:
%   y - n x 1 response
%   X - n x p design matrix
%   beta - p x 1 input
%   lambda - regularization parameter
%   b - shrinkage
% Output:
%   t - optimal stepsize
t = 1;
Gt = generalizedGrad(t,y,X,beta,lambda);
while gFunc(y, X, beta-t*Gt) > gFunc(y, X, beta) ...
        - t * grad(y,X,beta)' * Gt + t/2 * norm(Gt,2)^2
    t = t * b;
    Gt = generalizedGrad(t,y,X,beta,lambda);
end

function G = generalizedGrad(t,y,X,beta,lambda)
% Input:
%   y - n x 1 response
%   X - n x p design matrix
%   beta - p x 1 input
%   lambda - regularization parameter
% Output:
%   G - generalized gradient Gt(beta)
G = (beta - softThres(beta - t * grad(y,X,beta),t,lambda)) / t;

function g = gFunc(y,X,beta)
% Compute the cost function g(beta) = ||y-X*beta||^2
% Input:
%   y - n x 1 response
%   X - n x p design matrix
%   beta - input vector
% Output:
%   g - g(beta)
g = norm(y-X*beta,2).^2;

function y = logdet(A)
% compute log deteminant of A

U = chol(A);
y = 2*sum(log(diag(U)));

