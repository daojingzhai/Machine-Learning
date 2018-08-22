function [simplereturn] = simple_return(pricedata)
    % find log-return of the stock market given prices of stock everyday. 
    simplereturn = ones(length(pricedata)-1,1);
    for i=1:length(simplereturn)
        simplereturn(i) = (pricedata(i+1)-pricedata(i))/pricedata(i);
    end
end