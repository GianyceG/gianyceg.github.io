---
layout: default
---

# Project - Explorations of time series analysis techniques and forecasting methods on real-world applications

---

### AUTHOR

Gianyce Gesualdo Ortiz

### UPDATED

October 1st 2024
---

# Literature Reviews

My semester project is focused on the exploration of time series analysis, particularly using ARIMA/SARIMA and Bayesian models, to forecast and analyze the spread of COVID-19. Additionally, the project examines socioeconomic data to predict future economic growth. This part of the page will showcase the literature reviews I have completed throughout the semester, highlighting the models and techniques applied in my research.

## Literature Review, Repo Week 1 Link - "Prediction and Analysis of COVID-19 Daily New Cases and Cumulative Cases: Time Series Forecasting and Machine Learning Models."
[Link to GitHub Repository](Capstone_Paper_Review_Literature_Review__Week1.pdf)


## Literature Review, Repo Week 2 Link - 
[Link to GitHub Repository](https://github.com/GianyceG/gianyceg.github.io/blob/main/Capstone_Paper_Review_Literature_Review__2_%20(1).pdf)

## Literature Review, Repo Week 2 Paper 2 Link - 
[Link to GitHub Repository](https://github.com/GianyceG/gianyceg.github.io/blob/main/Capstone_Paper_Review_Literature_Review__3_%20(1).pdf)

## Literature Review, Repo Week 3 Link - 
[Link to GitHub Repository](https://github.com/GianyceG/gianyceg.github.io/blob/main/Capstone_Paper_Review_Literature_Review__4_%20(1).pdf)

# Introduction

# Explorations of time series analysis techniques and forecasting methods on Real-World Applications.


The use of time series analysis gained recognition since 1970 when the Box-Jenkins method popularized ARIMA models to find the best fit of different time series models based on past data.  HoIver, the concept of analyzing sequential data over time is not new. As the world increasingly embraces data-driven solutions to forecast future industry outcomes, time series analysis has become vital in numerous fields. This includes public health initiatives, such as those that arose in response to the COVID-19 pandemic, and economic studies, such as forecasting income distribution gaps. Both highlight the importance of time series in addressing real-world issues.

Anything that is observed sequentially over time is a time series. In the data, I will look at regular intervals of time. The data I analyze typically occur at regular intervals, allowing us to observe trends and seasonal patterns. Time series forecasting is a statistical method used to predict future values based on past results. The simplest models focus solely on the variable being forecast, ignoring external factors like marketing efforts or economic shifts, and instead extrapolating existing trends and seasonality. Specifically in this capstone, I will be exploring univariate models.

Currently, efforts to perfect the science of forecasting is important as the higher the accuracy of a time series based on past data and events, the more essential the tool becomes for modern organizations as accurate forecasts support decision-making across various timeframes—whether short-term, medium-term, or long-term, depending on the specific application. Key considerations in forecasting include seasonality, trends, and external fluctuations, which play a significant role in shaping predictions. These specifically will be explored more as I deal with the data sets of this project. These forecasts are integral for planning, resource allocation, and adjusting strategies in response to upcoming movements.

In this capstone, I will explore the Johns Hopkins CSSEGISandData COVID-19 Repository, which tracks global COVID-19 cases and Our World in Data (OWID), which provides comprehensive COVID-19 datasets across various indicators datasets to test the univariate models ARIMA/SARIMA and Bayesian STS. Using the same model structures, I will also be exploring the Forecasting Income Distribution Gaps in the US, using the US Census datasets. The goal of this exploration is to apply time series analysis to forecast the future spread of COVID-19, seeing which models are the strongest to do such a task, and also to predict the income distribution gaps in the U.S. using real-world datasets. These two different datasets and applications are to show the strength in forcasting in multiple real life situations. This is important in data science and statistics, particularly in public health and economic policy as creating accurate forecasts can allow pollicy makers a better direction on what should be applied and when. 

To achieve the goals of my capstone, ARIMA/SARIMA for capturing linear trends and seasonal effects, and Bayesian STS for incorporating uncertainty and providing probabilistic forecasts. The project will begin with an exploratory data analysis (EDA) to visualize the time series data, identify patterns, and check for seasonality or trends. After cleaning and preprocessing the data, I will test both models using python on the COVID-19 and income distribution datasets. The results will then be compared to assess which approach yields the most accurate and insightful forecasts. 

-Things I still want to add:
-References to definitions, where I got model ideas

# Methods

My project explores time series forecasting using two distinct datasets: one focused on COVID-19 data and the other on U.S. Census data. I will begin by explaining the methods I plan to apply to the COVID-19 dataset.

This part of the project uses time series forecasting on the COVID-19 dataset from the Johns Hopkins CSSEGISandData Repository and Our World in Data (OWID), with an emphasis on modeling the data using ARIMA/SARIMA and Bayesian Structural Time Series (BSTS). AR and ARIMA models are univariate (single-variable) time series methods without trend or seasonal components, while other time series models, such as SARIMA, can capture multivariate series with trends and seasonal patterns. Specifically, I will focus on ARIMA since it is effective for univariate modeling, which aligns with the goals of this project.

ARIMA is well-suited for forecasting because it applies mathematical techniques to smooth non-stationary time series data, making it easier to predict future values by analyzing historical patterns. These models estimate and extrapolate future states based on trends observed in past and present data. The use of ARIMA for univariate forecasting in this context is ideal for capturing the patterns in the COVID-19 dataset.


### ARIMA Model

The ARIMA model is defined by the parameters \( (p, d, q) \), where:
- \( p \) is the number of lag observations (autoregressive terms).
- \( d \) is the number of times the data must be differenced to make it stationary.
- \( q \) is the number of lagged forecast errors (moving average terms).

For a general ARIMA model, the equation is:

The ARIMA model can be represented as:

\[
Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \dots + \phi_p Y_{t-p} + \theta_1 \epsilon_{t-1} + \dots + \theta_q \epsilon_{t-q} + \epsilon_t
\]


Where:
- \( Y_t \) is the value of the time series at time \( t \) (in our case, confirmed COVID-19 cases).
- \( c \) is a constant (intercept).
- \( \phi_1, \dots, \phi_p \) are the autoregressive coefficients.
- \( \theta_1, \dots, \theta_q \) are the moving average coefficients.
- \( \epsilon_t \) is the white noise error term at time \( t \).


## References

**[1] Kulshreshtha, Vikas, and N. K. Garg. 2020.** "Predicting the New Cases of Coronavirus [COVID-19] in India by Using Time Series Analysis as Machine Learning Model in Python." *The Institution of Engineers (India)*.

**[2] Demongeot J, Oshinubi K, Rachdi M, Hobbad L, Alahiane M, Iggui S, Gaudart J, Ouassou I. 2021.** "The Application of ARIMA Model to Analyze COVID-19 Incidence Pattern in Several Countries." *J Math Comput Sci*. [https://www.scik.org/index.php/jmcs/article/view/6541](https://www.scik.org/index.php/jmcs/article/view/6541).

**[3] Wang, Yanding, et al. 2022.** "Prediction and Analysis of COVID-19 Daily New Cases and Cumulative Cases: Time Series Forecasting and Machine Learning Models." *BMC Infectious Diseases*, vol. 22, p. 495. [https://doi.org/10.1186/s12879-022-07472-6](https://doi.org/10.1186/s12879-022-07472-6).

**[4] Fotia, Pasquale, and Massimiliano Ferrara. 2023.** "A Different Approach for Causal Impact Analysis on Python with Bayesian Structural Time-Series and Bidirectional LSTM Models." *Atti della Accademia Peloritana dei Pericolanti - Classe di Scienze Fisiche, Matematiche e Naturali*, vol. 101, no. 2.

**[5] Nielsen, Michael. 2015.** *Neural Networks and Deep Learning*. Determination Press.

**[6] Ning, Yanrui, Hossein Kazemi, and Pejman Tahmasebi. 2022.** "A Comparative Machine Learning Study for Time Series Oil Production Forecasting: ARIMA, LSTM, and Prophet." *Computers & Geosciences*, vol. 164, 105126. Available online 6 May 2022. [https://doi.org/10.1016/j.cageo.2022.105126](https://doi.org/10.1016/j.cageo.2022.105126).



