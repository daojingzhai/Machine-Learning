In this work, I try to look into the distribution of stock price. 

Using data of Composite Index from Shanghai Stock Exchange (see 'SSE Composite Index_2013-2018.csv' in Folder 'Data'). I do investigation with simple stock return and log stock return. Frequency density distribution and Probability density distribution (fitted with Norm distribution) are plotted in the program. 

Based on my data, I caculate Skewness and Kurtosis of Simple return and Log return. Both of the two distributions have more outlier-prone than the norm distribution, i.e. heavy tail, while both have none-zero skewness. In fact, log return is worse than simple return considering the distance to norm distribution. 

It is easy to interpret the data: 
- Negative skewness: The left tail is longer; the mass of the distribution is concentrated on the right of the figure. It appreas as a right-leaning curve. It's common for a developing country, especially for China, which keeps high developing pace over the years.
- Kurtosis > 3: heavy tail owning to risks in the market. 
