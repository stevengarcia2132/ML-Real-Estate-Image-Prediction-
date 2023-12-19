# Introduction
The real estate market in Southern California is dynamic and intricate, influenced by a myriad of factors ranging from location and property features to economic trends. In this machine learning project, we delve into the realm of real estate data, employing a holistic approach by combining both tabular and image data to gain a comprehensive understanding of property values.

I was interested in seeing how well a model can predict the house price based on the image and a few other factors. I also thought it would be interesting to see how models would do without images and instead just use variables like number of bed, bath, square footage, and citi. 

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
I experimented with taking out the City variable but the model performed significantly worse without it. The R-squared lowered to 0.348 and the Mean Squared Error (Linear Regression without city) was 96084035154.39. So after this initial test, I decided to keep the categorical variable of city in the models going forward.

~Decision Tree: It works by recursively partitioning the input space into regions and assigning a label or value to each region. The decision tree structure is hierarchical and consists of nodes, branches, and leaves.
This model didn't perform as well as the linear regression model but was still better than most models. Its Mean Absolute Error (Decision Tree): is 133913.811 and the R-squared (Decision Tree): is 0.539. So far the linear regression model was best.


~KNN: The algorithm identifies the k-nearest data points for a new input.
The predicted value is the average (or weighted average) of the target values of the k-nearest neighbors.
This model performed the worst out of all the models I tried on the tabular data. The Mean Squared Error (KNN) was 98231556851.54, the Mean Absolute Error (KNN) was 230828.625 and the R-squared (KNN) was 0.33. The KNN model was significantly worse than the linear regression and decision tree. I think a big reason for this was that KNN models perform best with classification. I tried many different values of K but once I tried above 15 I noticed little difference. 

~Nueral Network: a computational model inspired by the structure and functioning of the human brain, consisting of interconnected nodes organized in layers that process and transform input data to produce output. The NN I used for the tabular was made of 3 layers. The beginning two layers were relu and the output layer was linear. I tried the Adam optimizer for 50 epochs and got pretty disappointing results. The R-squared (Neural Network) was 0.3290 and the Mean Absolute Error (Neural Network) was 229402.30851171244. I adjusted the layers a few times but continued to see little to no improvement in the metrics. This type of model seems too complex for the task. 


~Gradient Boosting: a powerful ensemble machine learning technique used for both classification and regression tasks. It builds a predictive model in the form of an ensemble of weak learners, typically decision trees.
The model I created used 150 estimators and had a learning rate of .1. Adjusting the learning rate made the biggest impact on the metrics but in the end it gave an R-squared (Gradient Boosting) slightly better than the NN with 0.387.

~AdaBoostRegressor: a machine learning algorithm that belongs to the family of ensemble learning methods and is specifically designed for regression tasks. AdaBoost (Adaptive Boosting) combines the predictions of multiple weak learners to create a strong predictive model. This was another disappointing model which produced an R-squared (AdaBoost) of  0.363. For this model, I used a  low learning rate like the last model and used 50 estimators. Overall this model seemed too complex for the data set and after adjusting hyperparameters I concluded it wasn't the best model

~Gradient Boosting: a powerful ensemble machine learning technique used for both classification and regression tasks. It builds a predictive model in the form of an ensemble of weak learners, typically decision trees.
The model I created used 150 estimators and had a learning rate of .1. Adjusting the learning rate made the biggest impact on the metrics but in the end it gave an R-squared (Gradient Boosting) slightly better than the NN with 0.387.

~Support Vector Machines for Regression (SVR): used to model the relationship between input features and a target variable by finding a hyperplane that captures the linear relationship in a high-dimensional space, allowing for non-linear relationships using kernel functions. This was the last model I used and I thought it would perform fairly well as it isn't very complex but sadly it did not. The MAE for SVR model was 269327.27931542526.27. Overall this model wasn't suited well to handle this data and because of that, it had an MAE higher than most of the other models. 

# Discussion of Best Model
So by far, the best model was the XG Boost model when predicting the house price based on the image. The XGboost had a Rsquared value of .921 and the Neural Network Rsquared was .314. The XGBoost had a great MAE value of 56945.714 compared to 231033.234375. The XGBoost model was much faster to compute compared to the NN. It took about 2.5 hours to run the NN with a batch size of 70 and 5 epochs. 
This is how the MSE of the neural network model changed over time.

![Figure](https://github.com/stevengarcia2132/ML-Real-Estate-Image-Prediction-/blob/main/NN-MSE.png)

The figure below is a similar graph but it is the mean absolute percentage error. 

![Figure](https://github.com/stevengarcia2132/ML-Real-Estate-Image-Prediction-/blob/main/trainValFig.png)


The XG boost model on the other hand took only a few minutes and was much more accurate in its prediction. One reason why the XGBoost model performed well was because XGBoost often requires less feature engineering. It can handle a mix of numeric and categorical features without extensive preprocessing.


Here is a look at the performances of each model. 
The figure below is the performance of the Neural Net

![Figure](https://github.com/stevengarcia2132/ML-Real-Estate-Image-Prediction-/blob/main/NNPerformance.png)

The figure below is the performance of the XGBoost model
![Figure](https://github.com/stevengarcia2132/ML-Real-Estate-Image-Prediction-/blob/main/XGPerformance.png)

As you can see the XGBoost was much more accurate which led to a better model. When Just looking at the tabular data there were a lot more models to compare but in the end, there wasn't one that was much better than the rest. The two that performed the best out of all the models are linear regression and decision tree. The linear regression model reported a higher R-squared but a lower MSE. The difference in MSE between the two was about fifteen percent which seemed weird when comparing the MSE's. In the end, I wanted the predicted values to be closer to the actual value and that's why the Linear Regression model is the one I would use going forward. 

# Conclusion
So to restate the XGBoost model performed much better compared to the Neural Network model. The XGBoost model is pre-trained and is well-equipped to handle image data without much tuning. The Neural Net was much slower and computation took about an hour and a half on average. This made trying different layers and activation formulas more difficult as it took a long time to see results. 

For the tabular data, the Linear regression model performed the best and was followed up by the Decision tree model. The linear regression model used a OneHotEncoder to handle the city variable as it is categorical. All the other models for the tabular data used this encoder but all reported higher MSE and lower R-squared than the regression model. It could be that most of the models tried on the tabular data were too complex and that there the relationships between the variables were not complex. 

As far as some possible future adjustments I would love to try some more complicated models for the image data set. There were some computation difficulties when handling the data set so not much testing was done with it. Pretrained models seemed to work the best with that data so trying [INSERT] could yield better results. I would be open to trying other types of Neural Networks with more layers if the computational power was available. I also think there could be some adjustments to the data. Having tons of cities does not seem that helpful. It would be better for the model if instead of having over 15,000 cities it had the county the city was listed in. I think this would drastically reduce the computation as there would much fewer values to compute for
