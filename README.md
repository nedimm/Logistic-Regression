# Logistic Regression
In this project we will implement logistic regression model to predict whether a student gets admitted into a university. The project is an exercise from the ["Machine Learning" course](https://www.coursera.org/learn/machine-learning/) from Andrew Ng.

The task is described as follows:
Suppose that you are the administrator of a university department and you want to determine each applicant’s chance of admission based on their results on two exams. You have historical data from previous applicants that you can use as a training set for logistic regression. For each training example, you have the applicant’s scores on two exams and the admissions decision.
Your task is to build a classification model that estimates an applicant’s probability of admission based the scores from those two exams. 

The implementation was done using [GNU Octave](https://www.gnu.org/software/octave/).

## Visualizing the data
Before starting to implement any learning algorithm, it is always good to visualize the data if possible. The figure below displays the historical data where the axes are the two exam scores, and the positive and negative examples are shown with different markers.
![Ex1-Ex2](https://i.imgur.com/DpjBPmU.png)

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
Now we will implement the cost function and gradient for logistic regression. The cost function in logistic regression is:
![cost function log regression](https://i.imgur.com/rvPF2Jd.png)
The gradient of the cost is a vector of the same length as ![theta](https://i.imgur.com/kGDVBc9.png) where the ![jth](https://i.imgur.com/TTKxOuh.png) element (for j = 0, 1,...,n) is defined as follows:
![gradient](https://i.imgur.com/3Kziad2.png)

