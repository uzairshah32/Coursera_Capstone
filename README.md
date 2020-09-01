## 1. Introduction
The world as a whole suffers due to car accidents, including the USA. National Highway Traffic Safety Administration of the USA suggests that the economical and societal harm from car accidents can cost up to $871 billion in a single year. According to 2017 WSDOT data, a car accident occurs every 4 minutes and a person dies due to a car crash every 20 hours in the state of Washington while Fatal crashes went from 508 in 2016 to 525 in 2017, resulting in the death of 555 people. The project aims to predict how severity of accidents can be reduced based on a few factors.

#### Stakeholders:
  - Public Development Authority of Seattle 
  - Car Drivers
  
## 2. Data
The dataset used for this project is based on car accidents which have taken place within the city of Seattle, Washington from the year 2004 to 2020. This data is regarding car accidents the severity of each car accidents along with the time and conditions under which each accident occurred. The data set used for this project can be found <a href="https://www.w3schools.com/">here!</a>. The model aims to predict the severity of an accident, considering that, the variable of Severity Code was in the form of 1 (Property Damage Only) and 2 (Injury Collision) which were encoded to the form of 0 (Property Damage Only) and 1 (Injury Collision). Following that, 0 was assigned to the element of each variable which can be the least probable cause of severe accident whereas a high number represented adverse condition which can lead to a higher accident severity. Whereas, there were unique values for every variable which were either ‘Other’ or ‘Unknown’, deleting those rows entirely would have led to a lot of loss of data which is not preferred.

<p align="center"><img src="https://github.com/uzairshah32/Coursera_Capstone/blob/master/Images/Variable%20Frequency.jpeg" width="700"/></p>

In order to deal with the issue of columns having a variation in frequency, arrays were made for each column which were encoded according to the original column and had equal proportion of elements as the original column. Then the arrays were imposed on the original columns in the positions which had ‘Other’ and ‘Unknown’ in them. This entire process of cleaning data led to a loss of almost 5000 rows which had redundant data, whereas other rows with unknown values were filled earlier.

#### Feature Selection
<p align="center"><img src="https://github.com/uzairshah32/Coursera_Capstone/blob/master/Images/Features.PNG" width="600"/></p>
 
## 3. Methodology

#### Exploratory Analysis
Considering that the feature set and the target variable are categorical variables with the likes of weather, road condition and light condition being an above level 2 categorical variables whose values are limited and usually based on a particular finite group whose correlation might depict a different image then what it actually is. Generally, considering the effect of these variables in car accidents are important hence these variables were selected. A few pictorial depictions of the dataset were made in order to better understand the data.


<p align="center"><img src="https://github.com/uzairshah32/Coursera_Capstone/blob/master/Images/2020-08-28%20(1).jpg" width="700"/></p>

The above figure illustrates, after data cleaning has taken place, the distribution of the target variables between Physical Injury and Property Damage Only. As it can be seen that the dataset is supervised but an unbalanced dataset where the distribution of the target variable is in almost 1:2 ratio in favor of property damage. It is very important to have a balanced dataset when using machine learning algorithms. Hence, SMOTE was used from imblearn library in order to balance the target variable in equal proportions in order to have an unbiased classification model which is trained on equal instances of both the elements under severity of accidents.

<p align="center"><img src="https://github.com/uzairshah32/Coursera_Capstone/blob/master/Images/2020-08-28.jpg" width="700"/></p>

As mentioned earlier, a number ‘0’ as an element of an independent variable is supposed to depict the least probable cause of a severe accident. The graph above is supposed to depict all the non-zero values within each independent variable of the model and can be seen as the frequency of adverse conditions under which accidents took place. The factor which had most number of accidents under adverse conditions was adverse weather conditions while adverse lighting condition had the second most number of accidents caused by it. The factors which contributed the least to an instance of an accident are over-speeding and the driver being under the influence.   
  
#### Machine Learning Models
- Logistic Regression
- Decision Tree Analysis
  
## 4. Results

#### Decision Tree Classification Report
<p align="center"><img src="https://github.com/uzairshah32/Coursera_Capstone/blob/master/Images/DT%20Classification%20Report.PNG" width="700"/></p>

#### Decision Tree Confusion Matrix
<p align="center"><img src="https://github.com/uzairshah32/Coursera_Capstone/blob/master/Images/DT%20Confusion%20Matrix.png" width="500"/></p>

#### Logistic Regression Classification Report
<p align="center"><img src="https://github.com/uzairshah32/Coursera_Capstone/blob/master/Images/LR%20Classification%20Report.PNG" width="700"/></p>

#### Logistic Regression Confusion Matrix
<p align="center"><img src="https://github.com/uzairshah32/Coursera_Capstone/blob/master/Images/LR%20Confussion%20Matrix.jpg" width="500"/></p>

## Sources
  - https://www.asirt.org/safe-travel/road-safety-facts/
  - https://www.nhtsa.gov/
  - https://wsdot.wa.gov/
