function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

XTheta = X*theta;
costTheta = theta;
costTheta(1,1) = 0;

J = (1/(2*m)) * sum((XTheta-y) .^ 2);
J += (lambda * sum(costTheta' .^ 2)) / (2*m);

grad = (1/m) * (X' * (XTheta-y));

for iter=2:size(theta,1)
  grad(iter) += (lambda/m) * theta(iter);    
end

% =========================================================================

grad = grad(:);

end
