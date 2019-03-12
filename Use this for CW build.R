setwd("c:/data")

dataset = read.csv("Housedata.csv")
dataset$date <- as.numeric(substr(dataset$date, 1,4))
dataset[1:21] <- lapply(dataset[1:21], as.numeric)
summary(dataset)
str(dataset)

#Use only variables with a certain vlue
#dataset = dataset[which(dataset$waterfront=='0'),]
#str(dataset)

library(caTools)
library(corrplot)
#checking correlation between variables
Housecor <- cor(dataset[1:21])
#plotting correlation
corrplot(Housecor, type="upper", order="hclust", tl.col="black", tl.srt=45)

#Split Dataset
set.seed(123)
split = sample.split(dataset$price, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

regressor = lm(formula = price ~ sqft_lot+sqft_above+sqft_basement+sqft_living+yr_built+sqft_lot+waterfront+view,   # price = dependant variable ., select all independant variables
               data = training_set)

summary(regressor)

#Regressor highlights problem with sqft_living so it is removed this may be as it is highly correlated to another variable

regressor = lm(formula = price ~ sqft_lot+sqft_above+sqft_basement+yr_built+sqft_lot+waterfront+view,   # price = dependant variable ., select all independant variables
               data = training_set)

summary(regressor)

#Significance is within the 5% range so we will predict some houses
y_pred = predict(regressor, newdata = test_set)

#Showing predictions as there are 1000's
y_pred[1000:1010]


#Removing variables set as least useful by correlation (waterfront,view) and using strong and medium

regressor = lm(formula = price ~ sqft_lot+sqft_above+sqft_basement+yr_built+sqft_lot,   # price = dependant variable ., select all independant variables
               data = training_set)

summary(regressor)

# Significance is within the 5% range so we will predict some houses
y_pred = predict(regressor, newdata = test_set)

#Showing predictions as there are 1000's
y_pred[1000:1010]

# Removing variables set as least useful by correlation (sqft_lot, yr_built), strong variables only used
regressor = lm(formula = price ~ sqft_lot+sqft_above+sqft_basement,   # price = dependant variable ., select all independant variables
               data = training_set)

summary(regressor)

# Significance is within the 5% range so we will predict some houses
y_pred = predict(regressor, newdata = test_set)

#Showing predictions as there are 1000's
y_pred[1000:1010]

# All in approach
regressor = lm(formula = price ~ .,
               data = training_set)

summary(regressor)

# Prediction using all in method
y_pred = predict(regressor, newdata = test_set)

#Showing only the first 15 predictions as there are 1000's
y_pred[1:10]



#-------------------------------------------------------------------------------------
#All in with backward elimination

setwd("c:/data")

dataset = read.csv("Housedata.csv")
dataset$date <- as.numeric(substr(dataset$date, 1,4))
dataset[1:21] <- lapply(dataset[1:21], as.numeric)
summary(dataset)
str(dataset)

#Use only variables with a certain vlue
#dataset = dataset[which(dataset$waterfront=='0'),]
#str(dataset)

library(caTools)
library(corrplot)

#Split Dataset
set.seed(123)
split = sample.split(dataset$price, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# price = dependant variable ., select all independant variables
regressor = lm(formula = price ~ .,   
               data = training_set)

summary(regressor)






