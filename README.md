# Car Accident Severity Machine Learning Analysis (Coursera Capstone)
This is an Applied Data Science Machine Learning Capstone Project for IBM Data Science Professional Certification.

## 1. Introduction
Worldwide, approximately 1.35 million people die in road crashes each year, on average 3,700 people lose their lives every day on the roads and an additional 20-50 million suffer non-fatal injuries, often resulting in long-term disabilities. 

The world as a whole suffers due to car accidents, including the USA. National Highway Traffic Safety Administration of the USA suggest that the economical and societal harm from car accidents can cost upto $871 billion in a single year. According to 2017 WSDOT data, a car accident occurs every 4 minutes and a person dies due to a car crash every 20 hours in the state of Washington while Fatal crashes went from 508 in 2016 to 525 in 2017, resulting in the death of 555 people.

#### Stakeholders:
  - State Department of Health
  - Insurance Companies
  - Emergency Services
  - Car Drivers
  
## 2. Data
The data being used for this machine learning model is regarding car accident severity in Seattle, Washington. The dataset has a total observations of 194673 with variation in number of observations for every attribute, hence an unbalanced dataset. The data type within the data set varies from being object, int64 and float64. The variables chosen from within the datase will be used to predict whether or not, if an accident occurs, an accident will lead to physical injury or property damage only. (https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Data-Collisions.csv)

#### Independant Variables
- INATTENTIONIND: Whether or not collision was due to inattention
- UNDERINFL: Whether or not a driver involved was under the influence of drugs or alcohol
- WEATHER: The weather conditions during the time of the collision
- ROADCOND: The condition of the road during the collision
- LIGHTCOND: The light conditions during the collision

#### Target Variable
- SEVERITYCODE: A code that corresponds to the severity of the collision

  
  
  
  
  
  
  
  
## Sources
  - https://www.asirt.org/safe-travel/road-safety-facts/
  - https://www.nhtsa.gov/
  - https://wsdot.wa.gov/
