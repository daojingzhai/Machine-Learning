clc;
clear;

%% data processing
load('Stockprice_daily.mat'); %stock price at the end of a day.
simplereturn = simple_return(stockprice_daily); % find simple return of the stock.
logreturn = log_return(simplereturn); % find log return of the stock.
intervalnum = 400;

%% simple return
figure;
[mu_simple, sigma_simple] = distri(simplereturn,intervalnum);
xlabel('Simple return');
ylabel('Probability density');
title('Simple return Probability density distributation');
fprintf('simple return mean = %f \n',var(simplereturn));
fprintf('simple return variance = %f \n',mean(simplereturn));
fprintf('simple return skewness = %f \n',skewness(simplereturn));
fprintf('simple return kurtosis = %f \n',kurtosis(simplereturn));

%% log return
figure;
[mu_log, nu_log]=distri(logreturn,intervalnum);
xlabel('Log return');
ylabel('Probability density')
title('Log return Probability density distributation');
fprintf('log return mean = %f \n',mean(logreturn));
fprintf('log return variance = %f \n',var(logreturn));
fprintf('log return skewness = %f \n',skewness(logreturn));
fprintf('log return kurtosis = %f \n',kurtosis(logreturn));