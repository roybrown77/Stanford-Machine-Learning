function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.
% Create New Figure

exam1 = X(:,1);
exam2 = X(:,2);
dataCount = size(y,1);
admitted = zeros(dataCount,2);
notAdmitted = zeros(dataCount,2);

for iter = 1:dataCount  
  yVal = y(iter);
  if yVal == 0
    notAdmitted(iter,1) = exam1(iter);
    notAdmitted(iter,2) = exam2(iter);
  else
    admitted(iter,1) = exam1(iter);
    admitted(iter,2) = exam2(iter);
  end
end

admittedExam1 = admitted(:,1);
admittedExam2 = admitted(:,2);

plot(admittedExam1,admittedExam2, 'k+','LineWidth', 2, 'MarkerSize', 7);

%figure; 
hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
% =========================================================================

notAdmittedExam1 = notAdmitted(:,1);
notAdmittedExam2 = notAdmitted(:,2);

plot(notAdmittedExam1,notAdmittedExam2, 'yo', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

% hold off;

end
