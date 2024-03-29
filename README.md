# HOUSING PRICE PREDICTIONS

### DATASET  DESCRIPTION:

The dataset contains descriptions and prices of 1000 houses in Ames, Iowa. The goal of the project was to predict housing prices from the 79 features used to describe the houses. 
<p>
<img src="figures/iowa-usa.png" width="700" height="350"/>
</p>
The features describing the houses included house condition, description of living room, bedrooms, basement, garage, and other spaces in the house, and other features describing the land around the house, neighborhood, street, etc.

<p float="left">
<img src="figures/floor%20plans.png" width="500" height="400"/>
<img src="figures/top-view.png" width="500" height="400"/>
</p>

The housing prices were distributed as following:

* The average house price was $ 177, 932
* Half of the houses had a price lower than $ 161, 625
* The most common house price was $ 135, 000
<p>
<img src="figures/house_prices_distribution.png" width="500" height="350"/>
</p>

### KEY TAKEAWAYS FROM THE STUDY

We found that overall material and finish of the house, rated between 1 (very poor) to 10 (very excellent), as well as the quality of the material on the exterior (Excellent, Good, Typical, Fair) were the most related to the house price. Houses with excellent quality material, finish and exterior were priced higher and vice versa.
<p float="left">
<img src="figures/Price_vs_overallQual.png" width="420" height="300"/>
<img src="figures/Price_vs_exteriorQual.png" width="420" height="300"/>
</p>

### HOUSE PRICE PREDICTIONS

We trained a model to predict house prices based on their features. We then tested the model on a test dataset of 100 houses. From the figure below, we see that our predicted prices are very close to the actual prices. The overall error in prediction is $16,400.

<p>
<img src="figures/prediction_comparison.png" width="500" height="375"/>
</p>

We also looked at the features that played the most important role in predicting the prices. 'Overall Material and Finish of House' and 'Living Room Area' were the two most important predictive features.

<p>
<img src="figures/feature_importance.png" width="700" height="400"/>
</p>
