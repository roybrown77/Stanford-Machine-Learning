function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(m, 1) X];

hiddenLayerCount = size(Theta1,1);

for iter=1:m
  for iter2=1:hiddenLayerCount
    hiddenLayer(iter,iter2) = sigmoid(a1(iter,:) * Theta1(iter2,:)');
  end;
end

hiddenLayer = [ones(m, 1) hiddenLayer];

for iter=1:m
  for iter2=1:num_labels
    outputLayer(iter,iter2) = sigmoid(hiddenLayer(iter,:) * Theta2(iter2,:)');    
  end
end

for iter=1:num_labels
  labels(iter) = iter;
end

for iter=1:m
  J += ( (-1 / m) * ( log(outputLayer(iter,:)) * (labels == y(iter))' + (log(1-outputLayer(iter,:)) * (1-(labels == y(iter))')) ) );  
end

thetaSummingVector = ones(size(Theta1(1,:),2),1);
thetaSummingVector(1,1) = 0;  

JTheta1 = 0;

for iter=1:hiddenLayerCount  
  JTheta1 += (lambda / (2 * m)) * ((Theta1(iter,:) .^2) * thetaSummingVector);
end

thetaSummingVector = ones(size(Theta2(1,:),2),1);
thetaSummingVector(1,1) = 0;  

JTheta2 = 0;

for iter=1:num_labels
  JTheta2 += (lambda / (2 * m)) * ((Theta2(iter,:) .^2) * thetaSummingVector);
end

J += JTheta1 + JTheta2;

% -------------------------------------------------------------

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
grad = [Theta1_grad(:) ; Theta2_grad(:);];

a2 = [ones(m,1) sigmoid(a1*Theta1')];
a3 = sigmoid(a2*Theta2');

h0 = a3;
Y = zeros(m, num_labels);

for i=1:m, Y(i,y(i)) = 1; end;
  J = sum(sum((-Y.*log(h0)) - (1-Y).*log(1-h0)))/m;
  regularisation = (lambda/(2*m)) * ((sum(sum(Theta1(:,2:end).^2))) + (sum(sum(Theta2(:,2:end).^2))));
  J = J + regularisation;
  Delta_1 = zeros(size(Theta1)); %4x5
  Delta_2 = zeros(size(Theta2)); %3x6

  for t=1:m
    %Step 1
    a1 = [1 X(t,:)]; %1x4
    a2 = [1 sigmoid(a1*Theta1')]; %1x6
    a3 = sigmoid(a2*Theta2'); %1x3
    
    %Step 2
    delta_3 = a3 - Y(t,:); %1x3
    
    %Step 3
    %size(Theta2) %3x6
    delta_2 = (delta_3*Theta2).*(a2.*(1-a2)); %1x6

    %Step 4
    Delta_1 = Delta_1 + delta_2(2:end)'*a1;
    Delta_2 = Delta_2 + delta_3'*a2;
  end

Theta1_grad = Delta_1/m + [zeros(size(Delta_1,1),1) (Theta1(:,2:end)*lambda)/m];
Theta2_grad = Delta_2/m + [zeros(size(Delta_2,1),1) (Theta2(:,2:end)*lambda)/m];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:);];

end