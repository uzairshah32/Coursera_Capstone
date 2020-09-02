## 1. Introduction
The world as a whole suffers due to car accidents, including the USA. National Highway Traffic Safety Administration of the USA suggests that the economical and societal harm from car accidents can cost up to *$871 billion* in a single year. According to 2017 WSDOT data, a car accident occurs every 4 minutes and a person dies due to a car crash every 20 hours in the state of Washington while Fatal crashes went from *508* in 2016 to *525* in 2017, resulting in the death of *555* people. The project aims to predict how severity of accidents can be reduced based on a few factors.

#### Stakeholders:
  - Public Development Authority of Seattle 
  - Car Drivers
  
## 2. Data
The dataset used for this project is based on car accidents which have taken place within the city of *Seattle, Washington* from the year *2004* to *2020*. This data is regarding the *severity of each car accidents* along with the time and conditions under which each accident occurred. The data set used for this project can be found **<a href="https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Data-Collisions.csv">here!</a>**. The model aims to predict the severity of an accident, considering that, the variable of Severity Code was in the form of *1* (Property Damage Only) and *2* (Physical Injury) which were encoded to the form of *0* (Property Damage Only) and *1* (Physical Injury). Following that, *0* was assigned to the element of each variable which can be the least probable cause of severe accident whereas a high number represented adverse condition which can lead to a higher accident severity. Whereas, there were unique values for every variable which were either *Other* or *Unknown*, deleting those rows entirely would have led to a lot of loss of data which is not preferred.

<p align="center"><img src="https://github.com/uzairshah32/Coursera_Capstone/blob/master/Images/Variable%20Frequency.jpeg" width="700"/></p>

In order to deal with the issue of columns having a variation in frequency, arrays were made for each column which were encoded according to the original column and had equal proportion of elements as the original column. Then the arrays were imposed on the original columns in the positions which had *Other* and *Unknown* in them. This entire process of cleaning data led to a loss of almost 5000 rows which had redundant data, whereas other rows with unknown values were filled earlier.

#### Feature Selection
<p align="center"><img src="https://github.com/uzairshah32/Coursera_Capstone/blob/master/Images/Features.PNG" width="600"/></p>
 
## 3. Methodology

#### Exploratory Analysis
Considering that the feature set and the target variable are categorical variables with the likes of weather, road condition and light condition being an above *level 2* categorical variables whose values are limited and usually based on a particular finite group whose correlation might depict a different image then what it actually is. Generally, considering the effect of these variables in car accidents are important hence these variables were selected. A few pictorial depictions of the dataset were made in order to better understand the data.


<p align="center"><img src="https://github.com/uzairshah32/Coursera_Capstone/blob/master/Images/2020-08-28%20(1).jpg" width="700"/></p>

The above figure illustrates, after data cleaning has taken place, the distribution of the target variables between Physical Injury and Property Damage Only. As it can be seen that the dataset is *supervised* but an *unbalanced* dataset where the distribution of the target variable is in almost 1:2 ratio in favor of property damage. It is very important to have a balanced dataset when using machine learning algorithms. Hence, SMOTE was used from imblearn library in order to balance the target variable in equal proportions in order to have an unbiased classification model which is trained on equal instances of both the elements under severity of accidents.

<p align="center"><img src="https://github.com/uzairshah32/Coursera_Capstone/blob/master/Images/2020-08-28.jpg" width="700"/></p>

As mentioned earlier, a number *0* as an element of an independent variable is supposed to depict the least probable cause of a severe accident. The graph above is supposed to depict all the non-zero values within each independent variable of the model and can be seen as the frequency of adverse conditions under which accidents took place. The factor which had most number of accidents under adverse conditions was adverse weather conditions while adverse lighting condition had the second most number of accidents caused by it. The factors which contributed the least to an instance of an accident are over-speeding and the driver being under the influence.   
  
#### Machine Learning Models chosen
- **Logistic Regression:** Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable
- **Decision Tree Analysis:** The Decision Tree Analysis breaks down a data set into smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes.
- **k-Nearest Neighbor:** K nearest neighbors is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure (based on distance)
  
## 4. Results
The results of each of the three models had variations among them, one worked very well at predicting the positives accurately while the other predicted the negatives better. 

### Decision Tree
The criterion chosen for the classifier was *entropy* and the max depth was *6*.

#### Decision Tree Classification Report
<p align="center"><img src="https://github.com/uzairshah32/Coursera_Capstone/blob/master/Images/DT%20Classification%20Report.PNG" width="700"/></p>

#### Decision Tree Confusion Matrix
<p align="center"><img src="https://github.com/uzairshah32/Coursera_Capstone/blob/master/Images/DT%20Confusion%20Matrix.png" width="500"/></p>

### Logistic Regression 
The C used for regularization strength was *0.01* whereas the solver used was *liblinear*.

#### Logistic Regression Classification Report
<p align="center"><img src="https://github.com/uzairshah32/Coursera_Capstone/blob/master/Images/LR%20Classification%20Report.PNG" width="700"/></p>

#### Logistic Regression Confusion Matrix
<p align="center"><img src="https://github.com/uzairshah32/Coursera_Capstone/blob/master/Images/LR%20Confussion%20Matrix.jpg" width="500"/></p>

### k-Nearest Neighbor
The best K, as shown below, for the model where the highest elbow bend exists is at *4*. 

#### Choosing the best K
<p align="center"><img src="https://github.com/uzairshah32/Coursera_Capstone/blob/master/Images/Best%20KNN%20value.jpeg" width="500"/></p>

#### k-Nearest Neighbor Classification Report
<p align="center"><img src="https://github.com/uzairshah32/Coursera_Capstone/blob/master/Images/KNN%20Classification.PNG" width="500"/></p>

## 5. Model Accuracy
- **Precision:** Precision refers to the percentage of results which are relevant, in simpler terms it can be seen as how many of the selected items from the model are relevant. Mathematically, it is calculated by dividing true positives by true positive and false positive
- **Recall:** Recall refers to the percentage of total relevant results correctly classified by the algorithm. In simpler terms, it tells how many relevant items were selected. It is calculated by dividing true positives by true positive and false negative
- **F1-Score:** f1-score is a measure of accuracy of the model, which is the harmonic mean of the model’s precision and recall. Perfect precision and recall is shown by the f1-score as 1, which is the highest value for the f1-score, whereas the lowest possible value is 0 which means that either precision or recall is 0
<p align="center"><img src="https://github.com/uzairshah32/Coursera_Capstone/blob/master/Images/Model%20Accuracy.PNG" width="500"/></p>

## 6. Conclusion
When comparing all the models by their *f1-scores*, *Precision* and *Recall*, we can have a clearer picture in terms of the accuracy of the three models individually as a whole and how well they perform for each output of the target variable. When comparing these scores, we can see that the f1-score is highest for k-Nearest Neighbor at *0.75*. However, later when we compare the precision and recall for each of the model, we can see that the k-Nearest Neighbor model performs poorly in the precision of *1* at *0.08*. The variance is too high for the model to be selected as a viable option. When looking at the other two models, we can see that the Decision Tree has a more balanced precision for *0* and *1*. Whereas, the Logistic Regression is more balanced when it comes to recall of *0* and *1*. Furthermore, the average f1-score of the two models are very close but for the Logistic Regression it is higher by *0.04*. It can be concluded that the both the models can be used side by side for the best performance. 

## 7. Recommendations
After assessing the data and the output of the Machine Learning models, a few recommendations can be made for the stakeholders. The developmental body for Seattle city can assess how much of these accidents have occurred in a place where road or light conditions were not ideal for that specific area and could launch development projects for those areas where most severe accidents take place in order to minimize the effects of these two factors. Whereas, the car drivers could also use this data to assess when to take extra precautions on the road under the given circumstances of light condition, road condition and weather, in order to avoid a severe accident, if any.

## Sources
- *https://www.macrotrends.net/cities/23140/seattle/population#:~:text=The%20current%20metro%20area%20population,a%201.2%25%20increase%20from%202017.*
- *https://www.seattletimes.com/seattle-news/data/housing-cars-or-housing-people-debate-rages-as-number-of-cars-in-seattle-hits-new-high/#:~:text=As%20of%202016%2C%20the%20total,are%20the%20number%20of%20cars.*
- *https://www.asirt.org/safe-travel/road-safety-facts/*
- *https://www.nhtsa.gov/*
- *https://wsdot.wa.gov/*
