function [mu, sigma] = distri(returndata,intervalnum)

%%  plot Probability density function
    ymin = min(returndata);
    ymax = max(returndata);
    interval_length = (ymax - ymin)/intervalnum;
    x = linspace(ymin,ymax,intervalnum);
    yy = hist(returndata,x);  %count the number in the intervals.
    yy = yy/length(returndata)/interval_length; %count the proportion of the intervals.
    bar(x,yy);
    hold on;
    
%%  plot Cumulative distribution function    
    %s=0;
    %for i=2:length(x)
    %  s=[s,trapz(x([1:i]),yy([1:i]))];
    %end
    %figure;
    %plot(x,s,x,s,'*')
    
%%  plot normfit distribution
    [mu,sigma] = normfit(returndata);
    fit=pdf('norm',returndata,mu,sigma);  
    plot(returndata,fit,'.') ;
    legend('frequency distribution','normfit');
end