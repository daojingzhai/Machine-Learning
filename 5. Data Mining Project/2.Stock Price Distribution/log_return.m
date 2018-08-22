function [logreturn] = log_return(simplereturn)
    % find log-return of the stock market given prices of stock everyday. 
    logreturn = log(1+simplereturn);
end