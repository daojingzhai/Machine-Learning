%% =========== Step 0: Initialization =============
clear ; 
close all; 
clc;
number_label = 5; 
data_dimension = 10; 

%% =========== Step 1: Loading Data =============
% In this part we load Data.

fprintf('\nLoad Data...\n');
train_feature = load('page_blocks_train_feature.txt');
train_label = load('page_blocks_train_label.txt');
test_feature = load('page_blocks_test_feature.txt');
test_label = load('page_blocks_test_label.txt');

%% ============ Step 2: Train Multi-class Classifiers Using Logistic Regression ============
% In this part we train classifiers.
% MCLR.m is multi-class logit, we use newton method.
% [all_beta] gives regression function.

Max_iter = 5000;
[all_beta] = zeros(number_label,data_dimension+1);
for i= 1:number_label
    X = train_feature;
    y = train_label==i;
    [X,y]=SMOTE(X, y);
    [beta] = MCLR(X,y, Max_iter);
    all_beta(i,:)= beta;
end 

%% ================ Step 3: Predict for One-Vs-All ================

fprintf('\nTest One-vs-All Logistic Regression Classifier...\n\n')
pred = predictOneVsRest(all_beta, test_feature);
for i=1:number_label
    [Accuracy] = accuracy(pred, test_label, i);
    fprintf('class %d precision rate: %f,',i,Accuracy(1,1))
    fprintf('recall rate: %f\n',Accuracy(2,1))
end
fprintf('Training Set Accuracy: %f\n', mean(double(pred == test_label)));

%% ================ Function: MCLR, Optimization using newton method ================
function [beta_row] = MCLR(X,y,Max_iter) 
%X: feature
%y: Label
%Max_iter: max interation
%c: classification
N = size(X,1);
p = size(X,2);
X_hat = [ones(N,1) X];
w = zeros(1,p)-1;
b = -1;
beta = [b w];
beta_update = [b+1 w+1];
epsilon  = 10^(-10);
iter_num = 0;
while (norm(beta_update - beta) > epsilon)&&(iter_num<Max_iter)
    beta = beta_update;
    grad = Grad(beta,X_hat,y);
    InvHess = invHess(beta,X_hat,y);
    beta_update = beta - grad*InvHess';
    iter_num = iter_num+1;
end
b = beta_update(1);
w = beta_update(2:end);
beta_row = [b,w];
end

%% ================ Function: Pr ================
function [p] = Sigmoid(beta,X_hat)
% p is a row vector
p =1./(1+exp(-beta*X_hat'));
end

%% ================ Function: Gradient ================
function [g] = Grad(beta,X_hat,y)
g = - sum(X_hat.*(y-Sigmoid(beta,X_hat)'));
end

%% ================ Function: Invhess martix ================
function [h] = invHess(beta,X_hat,y)
p = size(X_hat,2);
N = size(y);
h = zeros(p);
Prob = Sigmoid(beta,X_hat).*(1-Sigmoid(beta,X_hat));
for i = 1:N
    h = h + X_hat(i,:)'*X_hat(i,:)*Prob(i);
end
h = pinv(h);  
end
            
%% ================ Function: prdict and output ================
function p = predictOneVsRest(all_theta, test_feature)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters all_theta


m = size(test_feature, 1);
num_labels = size(all_theta, 2);

p = zeros(size(test_feature, 1), 1);

% Add ones to the X data matrix
test_feature = [ones(m, 1) test_feature];

C = sigmoid(test_feature*all_theta');
[M , p] = max(C , [] , 2);


end

%% ================ Function: sigmoid ================
function f = sigmoid (x)
    f = 1 ./ (1 + exp(-x));
end

%% ================ Function: find pression rate and recall rate ================
function [Accuracy] = accuracy(pred, test_label, i)
TP = 0; %True Positive
FP = 0; %False Positive
FN = 0; %False Negative
Accuracy = zeros(2,1);
for j = 1:length(test_label)
    if pred(j,1)==i
        if test_label(j,1)==i
            TP=TP+1;
        else
            FP=FP+1;
        end
    end
    if (test_label(j,1)==i) && (pred(j,1)~=i)
        FN=FN+1;
    end
  
Accuracy(1,1) = TP/(TP+FP); % Precision rate
Accuracy(2,1) = TP/(TP+FN); % Recall rate

end
end
%% ================ Function: SMOTE, augmentation the positve samples when they are much fewer than neg ================
function [X_augment,y_augment] = SMOTE(X,y)
% When the positive is much fewer than neg , we need this augmentation.
% the formal definition for skewed class is :<5% , and I will try to augment
% the data up to 10% or more.
pos = numel(y(y==1)); 
rate = pos./numel(y);
X_augment = X;
y_augment = y;
if rate > 0.05 %Donot need augment
    return;
end
while (rate<0.1)
    X_chosen = X_augment(y==1,:);
    N = 2; %twice the pos mark everytime.
    row = size(X_chosen,1); 
    K = 5; %nearst 5 neighors 
    for i = 1:row
        X_chosen_elim = X_chosen-X_chosen(i,:);
        dist = vecnorm(X_chosen_elim,2,2); %Euclidean distance
        [~,index] = sort(dist);%get the nearest index
        KNN = index(2:K+1);%remove the itself.
        sel = KNN(randi(K,N,1));
        cof = rand(N,1);
        X_new = X_chosen(i,:)+cof.*(X_chosen(sel,:)-X_chosen(i,:));
        X_augment = [X_augment;X_new];
        y_augment = [y_augment;ones(N,1)];
    end
        pos = numel(y_augment(y_augment==1)); 
        rate = pos./numel(y_augment);
end
end
