# Logistic Regression
In this project we will implement logistic regression model to predict whether a student gets admitted into a university. The project is an exercise from the ["Machine Learning" course](https://www.coursera.org/learn/machine-learning/) from Andrew Ng.

The task is described as follows:
Suppose that you are the administrator of a university department and you want to determine each applicant’s chance of admission based on their results on two exams. You have historical data from previous applicants that you can use as a training set for logistic regression (stored in the `ex2data1.txt` file). For each training example, you have the applicant’s scores on two exams and the admissions decision.
Your task is to build a classification model that estimates an applicant’s probability of admission based the scores from those two exams. 

The implementation was done using [GNU Octave](https://www.gnu.org/software/octave/). The start point is the `ex2.m` script and other functions are implemented in separate `*.m` files.

## Visualizing the data
Before starting to implement any learning algorithm, it is always good to visualize the data if possible. The figure below displays the historical data where the axes are the two exam scores, and the positive and negative examples are shown with different markers.

![Ex1-Ex2](https://i.imgur.com/DpjBPmU.png)
*Figure 1: Training data*

## Sigmoid function
Before we start with the actual cost function, we recall that the logistic regression hypthesis is defined as:

![log-reg-hypothesis](https://i.imgur.com/DmDgMAR.png)

where function g is the sigmoid function. Our task is to fit parameters ![theta](https://i.imgur.com/kGDVBc9.png) and the hypothesis representation above will give us the prediction.
The sigmoid function is defined as:

![sigmoid function](https://i.imgur.com/QuvjrRz.png)

Our first step is to implement this function so it can be called by the rest of our program.
```matlab
function g = sigmoid(z)
    % SIGMOID Compute sigmoid function
    % g = SIGMOID(z) computes the sigmoid of z.
    % z can be a matrix, vector or scalar
    g = 1 ./ (1 + exp(-z));
end
```
Testing the function on the command line gives us:

![testing sigmoid](https://i.imgur.com/7pUGnL2.png)

For large positive values of x, the sigmoid is close to 1, while for large negative values, the sigmoid is close to 0. Evaluating sigmoid(0) gives us exactly 0.5. This code works also with vectors and matrices. For a matrix, the function performs the sigmoid function on every element.

## Cost function and gradient
Now we will implement the cost function and gradient descent for logistic regression. The cost function in logistic regression is:

![cost function](https://i.imgur.com/rvPF2Jd.png)

The way we're going to minimize the cost function is using gradient descent. Here's the usual template for gradient descent where we repeatedly update each parameter by updating it as itself minus learning rate alpha times the derivative term.
![gradient-descent](https://i.imgur.com/R8mWMqa.png)

The gradient of the cost is a vector of the same length as ![theta](https://i.imgur.com/kGDVBc9.png) where the ![jth](https://i.imgur.com/TTKxOuh.png) element (for j = 0, 1,...,n) is defined as follows:
![gradient](https://i.imgur.com/3Kziad2.png)

This gradient looks identical to the linear regression gradient, the formula is actually different because linear and logistic regression have different definitions of hθ(x).
The implementation of the cost function looks like this:
```matlab
function [J, grad] = costFunction(theta, X, y)
  %COSTFUNCTION Compute cost and gradient for logistic regression
  %   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
  %   parameter for logistic regression and the gradient of the cost
  %   w.r.t. to the parameters.

  m = length(y); % number of training examples
  J = 0;
  grad = zeros(size(theta));
  J = (1 / m) * sum( -y'*log(sigmoid(X*theta)) - (1-y)'*log( 1 - sigmoid(X*theta)) );
  grad = (1 / m) * sum( X .* repmat((sigmoid(X*theta) - y), 1, size(X,2)) );
end
```

The output of the ex2.m shows now correct values:

![cmd1](https://i.imgur.com/jN3d70p.png)

## Optimizing using `fminunc`

Octave’s fminunc is an optimization solver that finds the minimum of an unconstrained function. For logistic regression, we want to optimize the cost function J(θ) with parameters θ.
For this purpose we will use `fminunc` to find the best parameters θ for the logistic regression cost function, given a fixed dataset (of X and y values). We will pass to `fminunc` the following inputs:
* The initial values of the parameters we are trying to optimize.
* A function that, when given the training set and a particular θ, computes the logistic regression cost and gradient with respect to θ for the dataset (X, y)
* Options for `fminunc`

```matlab
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
```
In this code snippet, we first defined the options to be used with fminunc. Specifically, we set the GradObj option to on, which tells fminunc that our function returns both the cost and the gradient. This allows fminunc to use the gradient when minimizing the function. Furthermore, we set the MaxIter option to 400, so that fminunc will run for at most 400 steps before it terminates.
To specify the actual function we are minimizing, we use a "short-hand" for specifying functions with the @(t) ( costFunction(t, X, y) ) . This creates a function, with argument t, which calls your costFunction. This allows us to wrap the costFunction for use with fminunc. If the costFunction was implemented correctly, fminunc will converge on the right optimization parameters and return the final values of the cost and θ. Notice that by using fminunc, we did not have to write any loops yourself, or set a learning rate like you did for gradient descent. This is all done by fminunc: we only needed to provide a function calculating the cost and the gradient.
Once fminunc completes, ex2.m will call our costFunction function using the optimal parameters of θ. 
We can see from the command promput output that the cost at theta is about 0.203. This final θ value will then be used to plot the decision boundary on the training data.

![decision boundary](https://i.imgur.com/PS5pWCZ.png)
*Figure 2: Traning data with Decision boundary*

## Evaluating logistic regression

After learning the parameters, we can use the model to predict whether a particular student will be admitted. For a student with an Exam 1 score of 45 and an Exam 2 score of 85, we should expect to see an admission probability of 0.776.
Another way to evaluate the quality of the parameters we have found is to see how well the learned model predicts on our training set. 

```matlab
function p = predict(theta, X)
  %PREDICT Predict whether the label is 0 or 1 using learned logistic 
  %regression parameters theta
  %   p = PREDICT(theta, X) computes the predictions for X using a 
  %   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

  m = size(X, 1); % Number of training examples
  p = sigmoid(X * theta)>=0.5 ;
end
```

The predict function will produce "1" or "0" predictions given a dataset and a learned parameter vector θ.
The `ex2.m` script will proceed to report the training accuracy of your classifier by computing the
percentage of examples it got correct.

![output](https://i.imgur.com/MVgYKtk.png)



