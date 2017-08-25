function w = fusedLassoSVM(X,y,t1,t2,varargin)
% solving the linear programming for fused-lasso SVM
% minimize_{w} \sum_i max(0,1 - y_i * x_i^T * w) 
% subject to: ||w||_1 < t_1
%             \sum_j |w_j - w_{j+1}| < t_2
%
% Input:
%   X - nxp data matrix
%   y - nx1 dependent variable
%   t1 - regularization parameter for l1 norm of weights
%   t2 - regularization parameter for l1 norm of difference
%   solver (optional) - default: MOSEK, recommended
% Output:
%   w - px1 weights
%
% @2017, Yuanning Li, ynli@cmu.edu

[n,p] = size(X);
% cost function
c = [ones(n,1);zeros(3*p-1,1)];
% lower and upper bounds
blc = [zeros(n,1);zeros(p,1);zeros(p,1);0;zeros(p-1,1);zeros(p-1,1);0;ones(n,1)];
buc = [inf(n,1); inf(p,1); inf(p,1); t1; inf(p-1,1); inf(p-1,1); t2; inf(n,1)];
% constraints
I1 = zeros(p-1,p);
for i = 1 : p-1
    I1(i,i) = 1;
    I1(i,i+1) = -1;
end
A = [eye(n), zeros(n,p), zeros(n,p), zeros(n,p-1);...
    zeros(p,n), eye(p), eye(p), zeros(p,p-1);...
    zeros(p,n), -eye(p), eye(p), zeros(p,p-1);...
    zeros(1,n), zeros(1,p), ones(1,p), zeros(1,p-1);...
    zeros(p-1,n), I1, zeros(p-1,p), eye(p-1);...
    zeros(p-1,n), -I1, zeros(p-1,p), eye(p-1);...
    zeros(1,n), zeros(1,p), zeros(1,p), ones(1,p-1);...
    eye(n), diag(y)*X, zeros(n,p), zeros(n,p-1)];

if strcmp(varargin{1}, 'MATLAB')
    % use MATLAB solver
    w = linprog(c, [A;-A], [buc;-blc]);
else
    % use MOSEK solver
    A = sparse(A);
    sol = msklpopt(c, A, blc, buc, [], []);
    w = sol.sol.itr.xx(n+1:n+p,1);
end

