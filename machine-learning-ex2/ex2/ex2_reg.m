%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the second part
%  of the exercise which covers regularization with logistic regression.
%
%  You will need to complete the following functions in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

plotData(X, y);

% Put some labels
hold on;

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('y = 1', 'y = 0')
hold off;


%% =========== Part 1: Regularized Logistic Regression ============
%  In this part, you are given a dataset with data points that are not
%  linearly separable. However, you would still like to use logistic
%  regression to classify the data points.
%
%  To do so, you introduce more features to use -- in particular, you add
%  polynomial features to our data matrix (similar to polynomial
%  regression).
%

% Add Polynomial Features

% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Compute and display cost and gradient
% with all-ones theta and lambda = 10
test_theta = ones(size(X,2),1);
[cost, grad] = costFunctionReg(test_theta, X, y, 10);

fprintf('\nCost at test theta (with lambda = 10): %f\n', cost);
fprintf('Expected cost (approx): 3.16\n');
fprintf('Gradient at test theta - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============= Part 2: Regularization and Accuracies =============
%  Optional Exercise:
%  In this part, you will get to try different values of lambda and
%  see how regularization affects the decision coundart
%
%  Try the following values of lambda (0, 1, 10, 100).
%
%  How does the decision boundary change when you vary lambda? How does
%  the training set accuracy vary?
%

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (with lambda = 1): 83.1 (approx)\n');




#{
lambdaValues=zeros(12,1);
predictions=zeros(12,1);
lambda=0;
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
  p = predict(theta, X);
lambdaValues(1)=lambda;
predictions(1)=mean(double(p == y)) * 100;
fprintf('Train Accuracy(lambda =0): %f\n', mean(double(p == y)) * 100);

lambda=0.001;
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
  p = predict(theta, X);
lambdaValues(2)=lambda;
predictions(2)=mean(double(p == y)) * 100;
fprintf('Train Accuracy(lambda =0.001): %f\n', mean(double(p == y)) * 100);

lambda=0.01;
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
  p = predict(theta, X);
lambdaValues(3)=lambda;
predictions(3)=mean(double(p == y)) * 100;
fprintf('Train Accuracy(lambda =0.01): %f\n', mean(double(p == y)) * 100);

lambda=0.1;
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
  p = predict(theta, X);
lambdaValues(4)=lambda;
predictions(4)=mean(double(p == y)) * 100;
fprintf('Train Accuracy(lambda =0.1): %f\n', mean(double(p == y)) * 100);


lambda=1;
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
  p = predict(theta, X);
lambdaValues(5)=lambda;
predictions(5)=mean(double(p == y)) * 100;
fprintf('Train Accuracy(lambda =1): %f\n', mean(double(p == y)) * 100);
lambda=2;
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
  p = predict(theta, X);
lambdaValues(6)=lambda;
predictions(6)=mean(double(p == y)) * 100;
fprintf('Train Accuracy(lambda =2): %f\n', mean(double(p == y)) * 100);
lambda=5;
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
  p = predict(theta, X);
lambdaValues(7)=lambda;
predictions(7)=mean(double(p == y)) * 100;
fprintf('Train Accuracy(lambda =5): %f\n', mean(double(p == y)) * 100);
lambda=10;
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
  p = predict(theta, X);
lambdaValues(8)=lambda;
predictions(8)=mean(double(p == y)) * 100;
fprintf('Train Accuracy(lambda =10): %f\n', mean(double(p == y)) * 100);
lambda=20;
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
  p = predict(theta, X);
lambdaValues(9)=lambda;
predictions(9)=mean(double(p == y)) * 100;
fprintf('Train Accuracy(lambda =20): %f\n', mean(double(p == y)) * 100);

lambda=30;
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
  p = predict(theta, X);
lambdaValues(10)=lambda;
predictions(10)=mean(double(p == y)) * 100;
fprintf('Train Accuracy(lambda =30): %f\n', mean(double(p == y)) * 100);

lambda=40;
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
  p = predict(theta, X);
lambdaValues(11)=lambda;
predictions(11)=mean(double(p == y)) * 100;
fprintf('Train Accuracy(lambda =40): %f\n', mean(double(p == y)) * 100);

lambda=100;
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
  p = predict(theta, X);
lambdaValues(12)=lambda;
predictions(12)=mean(double(p == y)) * 100;
fprintf('Train Accuracy(lambda =100): %f\n', mean(double(p == y)) * 100);
lambda=1000;
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
  p = predict(theta, X);
%lambdaValues(10)=lambda;
%predictions(10)=mean(double(p == y)) * 100;
fprintf('Train Accuracy(lambda =1000): %f\n', mean(double(p == y)) * 100);
lambda=10000;
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
  p = predict(theta, X);
%lambdaValues(11)=lambda;
%predictions(11)=mean(double(p == y)) * 100;
fprintf('Train Accuracy(lambda =10000): %f\n', mean(double(p == y)) * 100);
plot(lambdaValues, predictions, 'k+','LineWidth', 2, 'MarkerSize', 7);

title(sprintf('lambda vs accuracy'))

% Labels and Legend
xlabel('lambda')
ylabel('accuracy')
#}