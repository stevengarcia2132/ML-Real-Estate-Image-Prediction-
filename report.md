# Introduction
The real estate market in Southern California is dynamic and intricate, influenced by a myriad of factors ranging from location and property features to economic trends. In this machine learning project, we delve into the realm of real estate data, employing a holistic approach by combining both tabular and image data to gain a comprehensive understanding of property values.

I was interested in seeing how well a model can predict the house price based on the image and a few other factors. I also thought it would be interesting to see how models would do without image and instead just using variables like number of bed, bath, square footage, and citi. 

The data was found [here](https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal/data)

# EDA

Data exploration wasn't as useful as I wanted it to be. The City variable has too many categories to really consider but I was able to look at other factors of the data.

The prices seemed to be mostly normal however there were a few homes that were outliers. 
![Figure](https://github.com/stevengarcia2132/ML-Real-Estate-Image-Prediction-/blob/main/scatter_plot_sqft_vs_price.png)


This image helped me realize how big the data set is. 12 Bedrooms is an outlier which is why it has no spread. 
![Figure](https://github.com/stevengarcia2132/ML-Real-Estate-Image-Prediction-/blob/main/AveragePrice%20for%20Bedrooms.png)





# Overview of models tried
Below are the models I used for this image data. 

~Neural Network (NN): a computational model inspired by the structure and functioning of the human brain, consisting of interconnected nodes organized in layers that process and transform input data to produce output.
~XGBoost: an efficient open-source machine learning library that implements the gradient boosting framework, specifically designed for speed and performance, often used for classification and regression tasks.

These are the models I used for the tabular data
~Linear Regression: a statistical method used for modeling the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data. I ran this model with a OneHotEncoder on the City variable otherwise I would have to drop the variable. Surprisingly this was one of the best performing models. The R-squared was 0.6904 and the Mean Absolute Error was 142438.6490. Now this might not seem that good but compared to the more complicated models that were used, this one did great!
I experimented with taking out the City variable but the model performed significantly worse without it. The R-squared lowered to 0.348 and the Mean Squared Error (Linear Regression without city) was 96084035154.39. So after this initial test I decided to keep the categorical variable of city in the models going forward.

~Decision Tree: It works by recursively partitioning the input space into regions and assigning a label or value to each region. The decision tree structure is hierarchical and consists of nodes, branches, and leaves.
This model didn't perform as well as the linear regression model but was still better than most models. Its Mean Absolute Error (Decision Tree): 133913.811 and the R-squared (Decision Tree): 0.539. So far the linear regression model was best.


~KNN: The algorithm identifies the k-nearest data points for a new input.
The predicted value is the average (or weighted average) of the target values of the k-nearest neighbors. 


~Nueral Network:a computational model inspired by the structure and functioning of the human brain, consisting of interconnected nodes organized in layers that process and transform input data to produce output.
~Gradient Boosting: a powerful ensemble machine learning technique used for both classification and regression tasks. It builds a predictive model in the form of an ensemble of weak learners, typically decision trees.
~AdaBoostRegressor: a machine learning algorithm that belongs to the family of ensemble learning methods and is specifically designed for regression tasks. AdaBoost (Adaptive Boosting) combines the predictions of multiple weak learners to create a strong predictive model.
~Support Vector Machines for Regression (SVR): used to model the relationship between input features and a target variable by finding a hyperplane that captures the linear relationship in a high-dimensional space, allowing for non-linear relationships using kernel functions.

# Discussion of Model Selection



# Discussion of Best Model
So by far, the best model was the XG Boost model when predicting the house price based on the image. The XGboost had a Rsquared value of .921 and the Nueral Network Rsquared was .314. The XGBoost had a great MAE value of 56945.714 compared to 231033.234375. The XGBoost model was much faster to compute compared to the NN. It took about 2.5 hours to run the NN with a batch size of 70 and 5 epochs. The XG boost model on the other hand took only a few minutes and was much more accurate in its prediction. One reason why the XGBoost model performed well was because XGBoost often requires less feature engineering. It can handle a mix of numeric and categorical features without extensive preprocessing.


When Just looking at the tabular data there was a lot more models to compare but in the end there wasn't one that was much better than the rest.

Linear

# Conclusion and Next Steps
