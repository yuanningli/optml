function beta = groupLasso(y,X,lambda,costType,group,nGroup,stepsize,thres,nSteps)
% solving group lasso with logistic/least squares loss using
% proximal gradient descent
% Inputs:
%   X - nxp data matrix
%   y - nx1 dependent variable
%   lambda - regularization parameter
%   group - p x 1 group indicator vector 
%           (0 for beta0, 1 for 1st group, 2 for 2nd group,...)
%   nGroup - number of groups
%   stepsize - stepsize, use 'backtrack' for backtracking line search
%   costType - type of cost functions, leastsquare or logistic
%   thres - convergence threshold
%   nSteps - number of iteration steps
% Output:
%   beta - px1 weights
% 
% @ 2017 Yuanning Li, ynli@cmu.edu

if strcmp(stepsize,'backtrack')
    beta = zeros(size(X,2),1);
    fBack = zeros(nSteps,1);
    steps = 0;
    iStep = [];
    for k = 1 : nSteps
        [t,steps] = backtrack(y,X,beta,lambda,group,nGroup,costType,0.5,steps);
        beta = prox(beta - t * grad(y,X,beta,costType),t,lambda,group,nGroup);
        steps = steps + 1;
        iStep = [iStep,steps];
        fBack(steps) = costFunc(y,X,beta,lambda,group,nGroup,costType);
        if steps > 1 && norm(fBack(steps) - fBack(steps-1)) < thres
            break
        end
    end
% elseif stcmp(stepsize,'accelerate')
else
    t = stepsize;
    beta = zeros(size(X,2),1);
    fAcc = zeros(nSteps,1);
    beta1 = beta; % beta(k-1)
    beta2 = beta; % beta(k-2)
    for k = 1 : nSteps
        v = beta1 + (k-2)/(k+1)*(beta1 - beta2);
        beta = prox(v - t * grad(y,X,v,costType),t,lambda,group,nGroup);
        fAcc(k) = costFunc(y,X,beta,lambda,group,nGroup,costType);
        beta2 = beta1;
        beta1 = beta;
        if norm(beta2-beta1) < thres
            break
        end
    end
% else
%     t = stepsize;
%     beta = zeros(size(X,2),1);
%     f = zeros(nSteps,1);
%     for k = 1 : nSteps
%         beta = prox(beta - t * grad(y,X,beta,costType),t,lambda,group,nGroup);
%         f(k) = costFunc(y,X,beta,lambda,group,nGroup,costType);
%         if k > 1 && abs(f(k) - f(k-1)) < thres
%             break
%         end
%     end
end

function beta = prox(x,t,lambda,group,nGroup)
% compute the proximal operator for h(x) in group lasso, with t and lambda
% input:
% x - current input vector (p x 1)
% t - stepsize
% lambda - parameter
% group - index of which group each element belongs to
% output: 
% beta - proximal (p x 1)

wI = zeros(nGroup,1);
beta = zeros(size(x));
for i = 1 : nGroup
    idx = find(group == i);
    wI(i) = sqrt(length(idx));
    xI = x(idx,:);
    if norm(xI,2) >= lambda * t * wI(i)
        beta(idx) = (1 - t * lambda * wI(i) / norm(xI,2)) * xI;
    end
end
beta(1) = x(1);


function g = grad(y,X,beta,costType)
% gradiant of the least square function at beta
% input:
% y - n x 1 dependent variable
% X - n x p design matrix
% beta - previous vector beta (p x 1)
% output:
% g - gradient (p x 1)
switch costType
    case 'leastsquare'
        g = - 2 * X' * (y - X * beta);
    case 'logistic'
        g = -sum(X'*diag(y),2) + sum(X' * diag(exp(X*beta)./(1+exp(X*beta))), 2);
end

function f = costFunc(y,X,beta,lambda,group,nGroup,costType)
% Input:
% y - n x 1 response
% X - n x p design matrix
% beta - p x 1 input
% lambda - regularization parameter
% group - p x 1 group vector (0 for beta0)
% nGroup - number of groups
% costType - type of cost functions, leastsquare or logistic
% Output:
% f - f(beta) value
f = gFunc(y,X,beta,costType) + hFunc(beta,lambda,group,nGroup);

function g = gFunc(y,X,beta,costType)
% Compute the cost function g(beta)
% Input:
% y - n x 1 response
% X - n x p design matrix
% beta - input vector
% costType - type of cost functions, leastsquare or logistic
% Output:
% g - g(beta)
switch costType
    case 'leastsquare'
        g = norm(y-X*beta,2).^2;
    case 'logistic'
        g = -y'*(X*beta) + sum(log(1+exp(X*beta)));
end

function h = hFunc(beta,lambda,group,nGroup)
% Compute h(beta)
% Input
% beta - p x 1 input
% lambda - regularization parameter
% group - p x 1 group vector (0 for beta0)
% nGroup - number of groups
% Output
% h - h(beta) value
wI = zeros(nGroup,1);
h = 0;
for i = 1 : nGroup
    idx = find(group == i);
    wI(i) = sqrt(length(idx));
    xI = beta(idx,:);
    h = h + lambda * wI(i) * norm(xI,2);
end

function [t, steps] = backtrack(y,X,beta,lambda,group,nGroup,costType,b,crrntSteps)
% Input:
% y - n x 1 response
% X - n x p design matrix
% beta - p x 1 input
% lambda - regularization parameter
% group - p x 1 group vector (0 for beta0)
% nGroup - number of groups
% costType - type of cost functions, leastsquare or logistic
% b - shrinkage
% crrntSteps - current iteration step count
% Output:
% t - optimal stepsize
% steps - step count
t = 1;
Gt = generalizedGrad(t,y,X,beta,lambda,group,nGroup,costType);
while gFunc(y, X, beta-t*Gt, costType) > gFunc(y, X, beta, costType) ...
        - t * grad(y,X,beta,costType)' * Gt + t/2 * norm(Gt,2)^2
    t = t * b;
    Gt = generalizedGrad(t,y,X,beta,lambda,group,nGroup,costType);
    crrntSteps = crrntSteps + 1;
end
steps = crrntSteps;

function G = generalizedGrad(t,y,X,beta,lambda,group,nGroup,costType)
% Input:
% y - n x 1 response
% X - n x p design matrix
% beta - p x 1 input
% lambda - regularization parameter
% group - p x 1 group vector (0 for beta0)
% nGroup - number of groups
% costType - type of cost functions, leastsquare or logistic
%
% Output:
% G - generalized gradient Gt(beta)
G = (beta - prox(beta - t * grad(y,X,beta,costType),t,lambda,group,nGroup)) / t;
