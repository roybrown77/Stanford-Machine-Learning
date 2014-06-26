function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

CSigmaArray = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
count = size(CSigmaArray,2);
error = 1000;

for iter=1:count
  newC = CSigmaArray(1,iter);
  for iter2=1:count
    newSigma = CSigmaArray(1,iter2);
    
    fprintf('\nEvaluating x1: \n')
    disp(x1);
    fprintf('\nEvaluating x2: \n')
    disp(x2);
    
    model = svmTrain(X, y, newC, @(x1, x2) gaussianKernel(x1, x2, newSigma));
    
    predictions = svmPredict(model, Xval);
    newError = mean(double(predictions ~= yval))
    if (newError <= error)
      error = newError
      C = newC;
      sigma = newSigma;
    end
  end
end

% =========================================================================

end
