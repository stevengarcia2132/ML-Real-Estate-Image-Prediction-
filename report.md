# Introduction
The real estate market in Southern California is dynamic and intricate, influenced by a myriad of factors ranging from location and property features to economic trends. In this machine learning project, we delve into the realm of real estate data, employing a holistic approach by combining both tabular and image data to gain a comprehensive understanding of property values.

I was interested in seeing how well a model can predict the house price based on the image and a few other factors. I also thought it would be interesting to see how models would do without image and instead just using variables like number of bed, bath, square footage, and citi. 

The data was found [here](https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal/data)

# EDA

Data exploration wasn't as useful as a I wanted it to be. The City variable has too many categories to really consider it but i was able to look at other factors of the data.

![Figure](https://github.com/stevengarcia2132/ML-Real-Estate-Image-Prediction-/blob/main/scatter_plot_sqft_vs_price.png)



# Overview of models tried
Below are the models I used for this image data. 

~Neural Network (NN): a computational model inspired by the structure and functioning of the human brain, consisting of interconnected nodes organized in layers that process and transform input data to produce output.
~XGBoost: an efficient open-source machine learning library that implements the gradient boosting framework, specifically designed for speed and performance, often used for classification and regression tasks.

These are the models i used for the tabular data
~Linear Regression: a statistical method used for modeling the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data.
~Decision Tree: It works by recursively partitioning the input space into regions and assigning a label or value to each region. The decision tree structure is hierarchical and consists of nodes, branches, and leaves.
~KNN: The algorithm identifies the k-nearest data points for a new input.
The predicted value is the average (or weighted average) of the target values of the k-nearest neighbors.
~Nueral Network:a computational model inspired by the structure and functioning of the human brain, consisting of interconnected nodes organized in layers that process and transform input data to produce output.
~Gradient Boosting: a powerful ensemble machine learning technique used for both classification and regression tasks. It builds a predictive model in the form of an ensemble of weak learners, typically decision trees.
~AdaBoostRegressor: a machine learning algorithm that belongs to the family of ensemble learning methods and is specifically designed for regression tasks. AdaBoost (Adaptive Boosting) combines the predictions of multiple weak learners to create a strong predictive model.
~Support Vector Machines for Regression (SVR): used to model the relationship between input features and a target variable by finding a hyperplane that captures the linear relationship in a high-dimensional space, allowing for non-linear relationships using kernel functions.

# Discussion of Model Selection



# Discussion of Best Model
So the best model 

# Conclusion and Next Steps
