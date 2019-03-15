dataset <- read.csv('Artif.csv')

dataset$accident_index <- NULL
dataset$was_vehicle_left_hand_drive. <- NULL
dataset$journey_purpose_of_driver <- NULL
dataset$age_band_of_driver <- NULL
dataset$driver_imd_decile <- NULL
dataset$driver_home_area_type <- NULL
dataset$location_easting_osgr <- NULL
dataset$location_northing_osgr <- NULL
dataset$police_force <- NULL
dataset$local_authority_.district. <- NULL
dataset$local_authority_.highway. <- NULL
dataset$did_police_officer_attend_scene_of_accident <- NULL
dataset$age_band_of_casualty <- NULL
dataset$casualty_home_area_type <- NULL
dataset$casualty_imd_decile <- NULL
dataset$vehicle_imd_decile <- NULL


write.csv(dataset, file = "ArtifNew.csv")

dataset <- read.csv('ArtifNew.csv')
na_count <-sapply(dataset, function(y) sum(length(which(is.na(y)))))
na_count1 <- data.frame(na_count)

dataset$lsoa_of_accident_location <- NULL
dataset$casualty_reference <- NULL
dataset$casualty_class <- NULL
dataset$sex_of_casualty <- NULL
dataset$age_of_casualty <- NULL
dataset$casualty_severity <- NULL
dataset$pedestrian_location <- NULL
dataset$pedestrian_movement <- NULL
dataset$car_passenger <- NULL
dataset$bus_or_coach_passenger <- NULL
dataset$pedestrian_road_maintenance_worker <- NULL
dataset$casualty_type <- NULL

write.csv(dataset, file = "ArtifNew1.csv")

dataset <- read.csv('ArtifNew1.csv')
na_count <-sapply(dataset, function(y) sum(length(which(is.na(y)))))
na_count1 <- data.frame(na_count)

dataset$X.1 <- NULL
dataset$X <- NULL
dataset$date <- NULL
dataset$time <- NULL



dataset$weather_conditions <- gsub('9', '', dataset$weather_conditions) #Convert 9 to N/A
dataset$weather_conditions <- as.numeric(dataset$weather_conditions) #Convert column to a numeric
sum(is.na(dataset$weather_conditions)) #Count of how many rows have an N/A 
dataset <- dataset[!is.na(dataset$weather_conditions),] #Updating dataset by removing rows with an N/A

dataset$light_conditions <- gsub('7', '', dataset$light_conditions)
dataset$light_conditions <- as.numeric(dataset$light_conditions)
sum(is.na(dataset$light_conditions))
dataset <- dataset[!is.na(dataset$light_conditions),]

dataset$road_type <- gsub('9', '', dataset$road_type)
dataset$road_type <- as.numeric(dataset$road_type)
sum(is.na(dataset$road_type))
dataset <- dataset[!is.na(dataset$road_type),]

dataset$sex_of_driver <- gsub('3', '', dataset$sex_of_driver)
dataset$sex_of_driver <- as.numeric(dataset$sex_of_driver)
sum(is.na(dataset$sex_of_driver))
dataset <- dataset[!is.na(dataset$sex_of_driver),]

dataset <- dataset[!is.na(dataset$latitude),]
dataset <- dataset[!is.na(dataset$longitude),]

#-----------Converting -1 to N/A--------------------------------------------------------------------


dataset$vehicle_type <- gsub('-1', '', dataset$vehicle_type)
dataset$vehicle_type <- as.numeric(dataset$vehicle_type)

dataset$towing_and_articulation <- gsub('-1', '', dataset$towing_and_articulation)
dataset$towing_and_articulation <- as.numeric(dataset$towing_and_articulation)

dataset$vehicle_manoeuvre <- gsub('-1', '', dataset$vehicle_manoeuvre)
dataset$vehicle_manoeuvre <- as.numeric(dataset$vehicle_manoeuvre)

dataset$vehicle_location.restricted_lane <- gsub('-1', '', dataset$vehicle_location.restricted_lane)
dataset$vehicle_location.restricted_lane <- as.numeric(dataset$vehicle_location.restricted_lane)

dataset$junction_location <- gsub('-1', '', dataset$junction_location)
dataset$junction_location <- as.numeric(dataset$junction_location)

dataset$skidding_and_overturning <- gsub('-1', '', dataset$skidding_and_overturning)
dataset$skidding_and_overturning <- as.numeric(dataset$skidding_and_overturning)

dataset$hit_object_in_carriageway <- gsub('-1', '', dataset$hit_object_in_carriageway)
dataset$hit_object_in_carriageway <- as.numeric(dataset$hit_object_in_carriageway)

dataset$vehicle_leaving_carriageway <- gsub('-1', '', dataset$vehicle_leaving_carriageway)
dataset$vehicle_leaving_carriageway <- as.numeric(dataset$vehicle_leaving_carriageway)

dataset$hit_object_off_carriageway <- gsub('-1', '', dataset$hit_object_off_carriageway)
dataset$hit_object_off_carriageway <- as.numeric(dataset$hit_object_off_carriageway)

dataset$X1st_point_of_impact <- gsub('-1', '', dataset$X1st_point_of_impact)
dataset$X1st_point_of_impact <- as.numeric(dataset$X1st_point_of_impact)

dataset$sex_of_driver <- gsub('-1', '', dataset$sex_of_driver)
dataset$sex_of_driver <- as.numeric(dataset$sex_of_driver)

dataset$age_of_driver <- gsub('-1', '', dataset$age_of_driver)
dataset$age_of_driver <- as.numeric(dataset$age_of_driver)

dataset$engine_capacity_.cc. <- gsub('-1', '', dataset$engine_capacity_.cc.)
dataset$engine_capacity_.cc. <- as.numeric(dataset$engine_capacity_.cc.)

dataset$propulsion_code <- gsub('-1', '', dataset$propulsion_code)
dataset$propulsion_code <- as.numeric(dataset$propulsion_code)

dataset$age_of_vehicle <- gsub('-1', '', dataset$age_of_vehicle)
dataset$age_of_vehicle <- as.numeric(dataset$age_of_vehicle)

dataset$junction_detail <- gsub('-1', '', dataset$junction_detail)
dataset$junction_detail <- as.numeric(dataset$junction_detail)

dataset$junction_control <- gsub('-1', '', dataset$junction_control)
dataset$junction_control <- as.numeric(dataset$junction_control)

dataset$X2nd_road_class <- gsub('-1', '', dataset$X2nd_road_class)
dataset$X2nd_road_class <- as.numeric(dataset$X2nd_road_class)

dataset$X2nd_road_number <- gsub('-1', '', dataset$X2nd_road_number)
dataset$X2nd_road_number <- as.numeric(dataset$X2nd_road_number)

dataset$pedestrian_crossing.human_control <- gsub('-1', '', dataset$pedestrian_crossing.human_control)
dataset$pedestrian_crossing.human_control <- as.numeric(dataset$pedestrian_crossing.human_control)

dataset$pedestrian_crossing.physical_facilities <- gsub('-1', '', dataset$pedestrian_crossing.physical_facilities)
dataset$pedestrian_crossing.physical_facilities <- as.numeric(dataset$pedestrian_crossing.physical_facilities)

dataset$road_surface_conditions <- gsub('-1', '', dataset$road_surface_conditions)
dataset$road_surface_conditions <- as.numeric(dataset$road_surface_conditions)

dataset$special_conditions_at_site <- gsub('-1', '', dataset$special_conditions_at_site)
dataset$special_conditions_at_site <- as.numeric(dataset$special_conditions_at_site)

dataset$carriageway_hazards <- gsub('-1', '', dataset$carriageway_hazards)
dataset$carriageway_hazards <- as.numeric(dataset$carriageway_hazards)

na_count <-sapply(dataset, function(y) sum(length(which(is.na(y)))))
na_count1 <- data.frame(na_count)

dataset <- dataset[!is.na(dataset$junction_control),]
dataset <- dataset[!is.na(dataset$X2nd_road_class),]
dataset <- dataset[!is.na(dataset$age_of_vehicle),]
dataset <- dataset[!is.na(dataset$age_of_driver),]

#-----Replacing N/A with column median--------------------------------------------------------------
dataset$towing_and_articulation[is.na(dataset$towing_and_articulation)] <- median(dataset$towing_and_articulation, na.rm=TRUE)
dataset$vehicle_manoeuvre[is.na(dataset$vehicle_manoeuvre)] <- median(dataset$vehicle_manoeuvre, na.rm=TRUE)
dataset$vehicle_location.restricted_lane[is.na(dataset$vehicle_location.restricted_lane)] <- median(dataset$vehicle_location.restricted_lane, na.rm=TRUE)
dataset$junction_location[is.na(dataset$junction_location)] <- median(dataset$junction_location, na.rm=TRUE)
dataset$skidding_and_overturning[is.na(dataset$skidding_and_overturning)] <- median(dataset$skidding_and_overturning, na.rm=TRUE)
dataset$hit_object_in_carriageway[is.na(dataset$hit_object_in_carriageway)] <- median(dataset$hit_object_in_carriageway, na.rm=TRUE)
dataset$vehicle_leaving_carriageway[is.na(dataset$vehicle_leaving_carriageway)] <- median(dataset$vehicle_leaving_carriageway, na.rm=TRUE)
dataset$X1st_point_of_impact[is.na(dataset$X1st_point_of_impact)] <- median(dataset$X1st_point_of_impact, na.rm=TRUE)
dataset$sex_of_driver[is.na(dataset$sex_of_driver)] <- median(dataset$sex_of_driver, na.rm=TRUE)
dataset$engine_capacity_.cc.[is.na(dataset$engine_capacity_.cc.)] <- median(dataset$engine_capacity_.cc., na.rm=TRUE)
dataset$X2nd_road_number[is.na(dataset$X2nd_road_number)] <- median(dataset$X2nd_road_number, na.rm=TRUE)
dataset$towing_and_articulation[is.na(dataset$towing_and_articulation)] <- median(dataset$towing_and_articulation, na.rm=TRUE)
dataset$pedestrian_crossing.human_control[is.na(dataset$pedestrian_crossing.human_control)] <- median(dataset$pedestrian_crossing.human_control, na.rm=TRUE)
dataset$pedestrian_crossing.physical_facilities[is.na(dataset$pedestrian_crossing.physical_facilities)] <- median(dataset$pedestrian_crossing.physical_facilities, na.rm=TRUE)
dataset$road_surface_conditions[is.na(dataset$road_surface_conditions)] <- median(dataset$road_surface_conditions, na.rm=TRUE)
dataset$special_conditions_at_site[is.na(dataset$special_conditions_at_site)] <- median(dataset$special_conditions_at_site, na.rm=TRUE)
dataset$carriageway_hazards[is.na(dataset$carriageway_hazards)] <- median(dataset$carriageway_hazards, na.rm=TRUE)

na_count <-sapply(dataset, function(y) sum(length(which(is.na(y)))))
na_count1 <- data.frame(na_count)

#Saving Cleaned Dataset as a .CSV-----------------------------------------------------------------------------

#write.csv(dataset, file = "ArtifNew.csv")





#ANN-----------------------ANN---------------------------------------ANN----------------------------

# Converting accident severity to make this a 2 class classification problem
dataset$accident_severity <- gsub("1", "0", dataset$accident_severity)
dataset$accident_severity <- gsub("2", "0", dataset$accident_severity)
dataset$accident_severity <- gsub("3", "1", dataset$accident_severity)
dataset$accident_severity <- as.numeric(dataset$accident_severity)

#Setting all columns to Numeric
dataset[1:40] <- lapply(dataset[1:40], as.numeric)


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$accident_severity, SplitRatio = 0.80)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-40] = scale(training_set[-40])
test_set[-40] = scale(test_set[-40])

# Fitting ANN to the Training set
install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)
classifier = h2o.deeplearning(y = 'accident_severity',
                              training_frame = as.h2o(training_set),
                              activation = 'Rectifier',
                              hidden = c(20,20),
                              epochs = 100,
                              train_samples_per_iteration = -2)


# Predicting the Test set results
prob_pred = h2o.predict(classifier, newdata = as.h2o(test_set[-40]))
y_pred = (prob_pred > 0.5)
y_pred = as.vector(y_pred)

# Making the Confusion Matrix
cm = table(test_set[, 40], y_pred)
cm



#-----XGBoost-------------------------------------------------------------------------------------------------------------------------------


# Converting accident severity to make this a 2 class classification problem
dataset$accident_severity <- gsub("1", "0", dataset$accident_severity)
dataset$accident_severity <- gsub("2", "0", dataset$accident_severity)
dataset$accident_severity <- gsub("3", "1", dataset$accident_severity)
dataset$accident_severity <- as.numeric(dataset$accident_severity)


#Setting all columns to Numeric
dataset[1:40] <- lapply(dataset[1:40], as.numeric)



# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$accident_severity, SplitRatio = 0.80)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling not needed by XGBoost
#training_set[-40] = scale(training_set[-40])
#test_set[-40] = scale(test_set[-40])


#Fitting XGBoost to the training set
install.packages('xgboost')
library(xgboost)
classifier = xgboost(data = as.matrix(training_set[-40]), label = training_set$accident_severity, nrounds = 150)

#Applying K-fold cross validation
#install.packages('caret')
library(caret)
folds = createFolds(training_set$accident_severity, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[-x, ]
  classifier = xgboost(data = as.matrix(training_set[-40]), label = training_set$accident_severity, nrounds = 150)
  y_pred = predict(classifier, newdata = as.matrix(test_fold[-40]))
  y_pred = (y_pred >= 0.5)
  cm = table(test_fold[, 40], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})

accuracy = mean(as.numeric(cv))



install.packages('DiagrammeR')
library(DiagrammeR)

# View only the first tree in the XGBoost model
xgb.plot.tree(model = classifier, n_first_tree = 1)







#-Kernel SVM----------------------------------------------------------------------------------------------------------------------------


# Classification template

# Importing the dataset
# Imported already dataset <- read.csv('Artif.csv')

# Converting accident severity to make this a 2 class classification problem
dataset$accident_severity <- gsub("1", "0", dataset$accident_severity)
dataset$accident_severity <- gsub("2", "0", dataset$accident_severity)
dataset$accident_severity <- gsub("3", "1", dataset$accident_severity)
dataset$accident_severity <- as.numeric(dataset$accident_severity)


#Setting all columns to Numeric
dataset[1:40] <- lapply(dataset[1:40], as.numeric)

# Encoding the target feature as factor
dataset$accident_severity = factor(dataset$accident_severity, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$accident_severity, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-40] = scale(training_set[-40]) 
test_set[-40] = scale(test_set[-40])

# Fitting Kernel SVM to the Training set
install.packages('e1071')
library(e1071)
classifier = svm(formula = accident_severity ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-40])

# Making the Confusion Matrix
cm = table(test_set[, 40], y_pred)
cm




# Random Forrest----------------------------------------------------------------------------------------------------------------------------------------

#Model does not like these columns



# Converting accident severity to make this a 2 class classification problem
dataset$accident_severity <- gsub("1", "0", dataset$accident_severity)
dataset$accident_severity <- gsub("2", "0", dataset$accident_severity)
dataset$accident_severity <- gsub("3", "1", dataset$accident_severity)
dataset$accident_severity <- as.numeric(dataset$accident_severity)

#Setting all columns to Numeric
dataset[1:40] <- lapply(dataset[1:40], as.numeric)

# Encoding the target feature as factor
dataset$accident_severity = factor(dataset$accident_severity, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$accident_severity, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-40] = scale(training_set[-40])
test_set[-40] = scale(test_set[-40])

# Fitting Random Forrest Classification to the Training set
install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-40],
                          y = training_set$accident_severity,
                          ntree = 25)


# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-40])

# Making the Confusion Matrix
cm = table(test_set[, 40], y_pred)
cm




#-Arima Model------------------------------------------------------------------------------------------------------------------

# loading needed packages----------------------------------------------------------------------------------------------

library(dplyr) # Allows filtering and manipulation of data
library(ggplot2) # High quality graphs etc.
library(plotly) # enables interactive graphs
library(lubridate) # allows manipulation of time 
library(zoo) # manipulation functions
library(forecast) # time-series forecasting
library(tseries) # time-series analysis

options(stringsAsFactors = FALSE)

# Joining accidents 2005-2014 to 2015-------------------------------------------------------------------------------

newdata1 = rbind(read.csv("Accidents0514.csv") %>%
                   rename_("Accident_Index" ="ï..Accident_Index"),
                 read.csv("Accidents_2015.csv")) %>%
  mutate(Date=as.POSIXct(Date, format="%d/%m/%Y"))

# # Joining casualties 2005-2014 to 2015----------------------------------------------------------------------------

newdata2= rbind(read.csv("Casualties0514.csv") %>%
                  rename_("Accident_Index" ="ï..Accident_Index"),
                read.csv("Casualties_2015.csv") %>%
                  select(-Casualty_IMD_Decile))

# # Joining vehicles 2005-2014 to 2015------------------------------------------------------------------------------

newdata3= rbind(read.csv("Vehicles0514.csv") %>%
                  rename_("Accident_Index" ="ï..Accident_Index"),
                read.csv("Vehicles_2015.csv") %>%
                  select(-Vehicle_IMD_Decile))

# Viewing the dataset structure--------------------------------------------------------------------------------------

View_newdata=as.data.frame(t(sapply(list(newdata1,newdata2,newdata3),function(x){
  c(length(unique(x$Accident_Index)),
    length(x),
    nrow(x))
})))

colnames(View_newdata)=c("# Accident Count", "# Column Count","# Row Count")
rownames(View_newdata)=c("Accidents","Casualties","Vehicles")
View_newdata

# Splitting the drivers by gender-------------------------------------------------------------------------------

newdata3 %>% group_by(Sex_of_Driver) %>% summarize(num_accs=n()) %>% 
  mutate(Sex_of_Driver=c("Unknown","Male","Female","Unknown")) %>% 
  mutate(prop=paste(round(100*num_accs/sum(num_accs),2),"%"))

#Assigning driver gender to a subset

driver_gender <- newdata3 %>% group_by(Sex_of_Driver) %>% summarize(num_accs=n()) %>% 
  mutate(Sex_of_Driver=c("Unknown","Male","Female","Unknown")) %>% 
  mutate(prop=paste(round(100*num_accs/sum(num_accs),2),"%"))

p <-ggplot(driver_gender, aes(Sex_of_Driver, num_accs))
p +geom_bar(stat = "identity", aes(fill = Sex_of_Driver), position = "dodge") +
  xlab("Gender") + 
  ylab("Count") +
  ggtitle("Gender of drivers in accidents") +
  theme_bw()





# Viewing road accidents based on day of the week--------------------------------------------------------------

newdata1 %>% group_by(Day_of_Week) %>% summarize(num_accs=n()) %>% 
  mutate(Day_of_Week=c("Sun","Mon","Tue","Wed",
                       "Thu","Fri","Sat")) %>%
  mutate(prop=paste(round(100*num_accs/sum(num_accs),2),"%"))

#Histogram showing accidents per day of the week
qplot(newdata1$Day_of_Week,
      geom="histogram",
      binwidth = 0.5,  
      main = "Accidents based on day of week", 
      xlab = "Day", 
      ylab = "Number of accidents",
      fill=I("blue"), 
      col=I("red"), 
      alpha=I(.2))


#Viewing accidents based on the hour of each day of the week----------------------------------------------------

daily_data <- mutate(newdata1,day=weekdays(Date),hour=substring(Time,1,2)) %>% 
  arrange(Day_of_Week) %>%
  group_by(day, hour) %>% summarize(num_accs=n()) %>% 
  mutate(prop=round(100*num_accs/sum(num_accs), 1)) %>%
  filter(hour!="") 

#Plotting accidents based on hour and day of week----------------------------------------------------------------

f_axi <- function(plottitle,size=18,colour="black",
                  font = "Arial, sans-serif"){
  list(
    title = plottitle,
    titlefont = list(
      family = font,
      size = size,
      color = colour))}
plot_ly(daily_data %>% 
          group_by(day, hour) %>% summarize(tot=sum(num_accs)) %>% 
          mutate(prop=round(100*tot/sum(tot), 1)),
        x=~hour,y=~prop, color =~day, type = "scatter", mode = "lines") %>%
  add_trace(data=daily_data %>% group_by(hour) %>% 
              summarize(tot=sum(num_accs)) %>% 
              mutate(prop=round(100*tot/sum(tot), 1)),
            x=~hour,y=~prop,name="All Days",
            line=list(width=3)) %>% 
  layout(xaxis=f_axi("Hour"),yaxis=f_axi("Accident Proportion in Percentage"), 
         title="Accident Proportion in Percentage by Hour. Years 2005-2015") %>%
  layout(legend = list(x = 0.02, y = 0.99,font=list(size=14)))

#Histogram showing Accidents based on Hour and Day

#Change the order of the Days, on graph
daily_data$day <-factor(daily_data$day, 
                        levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
                                   "Sunday"))

p <-ggplot(daily_data, aes(hour, num_accs))
p +geom_bar(stat = "identity", aes(fill = day), position = "dodge") +
  xlab("Hour of Day") + 
  ylab("Accident Count") +
  ggtitle("Accidents based on Hour and Day") +
  theme_bw()



#Cleaning and reformatting days by year and month i.e. July 2005-----------------------------------------------------

yea_mon <- newdata1 %>% group_by(as.yearmon(Date, format="%d/%m/%Y")) %>% summarize(num_accs=n())
colnames(yea_mon)[1]="YearMonth"

#Plotting accidents based on month-----------------------------------------------------------------------------------

li_li <- list()
for(i in 1:length(unique(year(newdata1$Date)))){
  li_li[[i]]=list(type      = "line",
                  line      = list(color = "black", dash="dashdot"),
                  opacity   = 0.3,
                  x0        = unique(year(newdata1$Date))[i],
                  x1        = unique(year(newdata1$Date))[i],
                  xref      = "x",
                  y0        = min(yea_mon$num_accs),
                  y1        = max(yea_mon$num_accs),
                  yref      = "y")
}

plot_ly(yea_mon,x = ~YearMonth, y = ~num_accs, type = "scatter", 
        mode = "lines", text=sapply(yea_mon$YearMonth,toString),
        hoverinfo="text+y") %>% layout(shapes=li_li) %>%  
  layout(xaxis=f_axi("Year"),yaxis=f_axi("Accident Count"),
         title="Monthly Accidents (2005-2015)")

#Not used, extracts month from date
#mont_data <- mutate(newdata1,Mont=month(Date)) %>% 
#  arrange(Mont) %>%
#  group_by(Mont) %>% summarize(num_accs=n()) %>% 
#  mutate(prop=round(100*num_accs/sum(num_accs), 1)) %>%
#  filter(Mont!="") 


# Setting log for multiplicative model--------------------------------------------------------------------------------

decom_log <- stl(ts(log(yea_mon$num_accs),frequency = 12,start=2005),s.window = "periodic")
decom_log$time.series <- exp(decom_log$time.series)


#Graph based on decomp of accident-------------------------------------------------------------------------------------

dec_plot <- function(time_series){
  timeseries_plot <- as.data.frame(time_series$time.series) %>% 
    mutate(date = as.yearmon(time(time_series$time.series))) %>% 
    tidyr::gather(variable, value, -date) %>% transform(id = as.integer(factor(variable)))
  seas_plotly <- plot_ly(filter(timeseries_plot, variable == "seasonal"),
                         x = ~date, y = ~value, colors = "Dark2", name="season", 
                         type="scatter", mode="lines",
                         text=unique(sapply(timeseries_plot$date,toString)),hoverinfo="text + y") %>%
    layout(xaxis=list(title="", showticklabels = FALSE))
  remain_plotly <- plot_ly(filter(timeseries_plot, variable == "remainder"),
                           x = ~date, y = ~value, colors = "Dark2", name="noise", 
                           type="scatter", mode = "lines",
                           text=unique(sapply(timeseries_plot$date,toString)),hoverinfo="text + y") %>%
    add_trace(x = ~date, y= 1,mode = "lines",showlegend=FALSE,line = list(
      color = "gray",
      dash = "dashed"                                
    )) %>%
    layout(xaxis=list(title=""))
  plot_tren <- plot_ly(filter(timeseries_plot, variable == "trend"),
                       x = ~date, y = ~value, colors = "Dark2", name="trend", 
                       type="scatter", mode="lines",
                       text=unique(sapply(timeseries_plot$date,toString)),hoverinfo="text + y") %>%
    layout(xaxis=list(title="", showticklabels = FALSE))
  plot_dat <- plot_ly(group_by(timeseries_plot,date) %>% summarize(value = prod(value)),
                      x = ~date, y = ~value, colors = "Dark2", name="data", 
                      type="scatter", mode="lines",
                      text=unique(sapply(timeseries_plot$date,toString)),hoverinfo="text + y") %>%
    layout(title = "Multiplicative Model",
           xaxis=list(title="", showticklabels = FALSE))
  subplot(list(plot_dat, seas_plotly, plot_tren, remain_plotly),nrows=4) 
}
dec_plot(decom_log) %>% 
  layout(legend = list(font=list(size=14)))





# fitting ARIMA model to data based on road accidents-----------------------------------------------------------------


fit_arim <- auto.arima(ts(yea_mon$num_accs,frequency = 12,start=2005),
                       allowdrift = TRUE, approximation=FALSE)
fit_arim


fore_acc <- forecast(fit_arim, h=12)

#Plotting the ARIMA model for forecasting---------------------------------------------------------------------------


plot_ly(yea_mon, x=~YearMonth,y=~num_accs,type="scatter", mode="lines",name="Observed",
        text=~sapply(YearMonth,toString),hoverinfo="text+y+name") %>% 
  add_trace(x=c(max(yea_mon$YearMonth)+seq(1/12,1,1/12), 
                max(yea_mon$YearMonth)+seq(1,1/12,-1/12)),
            y=c(fore_acc$lower[,2],rev(fore_acc$upper[,2])),name="95% Confidence",
            fill="toself", hoveron = "points",
            text=c(sapply(max(yea_mon$YearMonth)+seq(1/12,1,1/12),toString),
                   sapply(max(yea_mon$YearMonth)+seq(1,1/12,-1/12),toString)), 
            hoverinfo="text+y+name", hoveron = "points") %>%
  add_trace(x=c(max(yea_mon$YearMonth)+seq(1/12,1,1/12), 
                max(yea_mon$YearMonth)+seq(1,1/12,-1/12)),
            y=c(fore_acc$lower[,1],rev(fore_acc$upper[,1])),name="80% Confidence",
            fill="toself", hoveron = "points",
            text=c(sapply(max(yea_mon$YearMonth)+seq(1/12,1,1/12),toString),
                   sapply(max(yea_mon$YearMonth)+seq(1,1/12,-1/12),toString)), 
            hoverinfo="text+y+name", hoveron = "points")  %>%
  add_trace(x=c(max(yea_mon$YearMonth)+seq(1/12,1,1/12)),
            y=as.vector(fore_acc$mean),name="Mean Prediction",
            text=sapply(max(yea_mon$YearMonth)+seq(1/12,1,1/12),toString), 
            hoverinfo="text+y+name", hoveron = "points") %>%
  add_trace(x=~YearMonth,
            y=as.vector(fore_acc$fitted),name="Model",
            text=~sapply(YearMonth,toString),hoverinfo="text+y+name",
            line = list(color = "#A9A9A9", dash = "dashed")) %>%
  layout(xaxis=f_axi("Year"),yaxis=f_axi("Number of Accidents"), 
         title="Number of Road Accidents By Month (2005-2016)") %>%
  layout(legend = list(x = 0.8, y = 0.99, font=list(size=14)))

#-----------------------------------------------------------------------------------------------------------------------


#Plotting bar graphs to visualise accidents----------------------------------------------------------------------------



#Plotting a bar graph to visualise accidents per hour-------------------------------------------------------------------



#number of accidents for each hour--------------------------------------------------------------------------------------
#Same daily data from inside previous code
daily_data <- mutate(newdata1,day=weekdays(Date),hour=substring(Time,1,2)) %>% 
  arrange(Day_of_Week) %>%
  group_by(day, hour) %>% summarize(num_accs=n()) %>% 
  mutate(prop=round(100*num_accs/sum(num_accs), 1)) %>%
  filter(hour!="") 



#Change the order of the Days, on graph
daily_data$day <-factor(daily_data$day, 
                        levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
                                   "Sunday"))

p <-ggplot(daily_data, aes(hour, num_accs))
p +geom_bar(stat = "identity", aes(fill = day), position = "dodge") +
  xlab("Hour of Day") + 
  ylab("Accident Count") +
  ggtitle("Accidents based on Hour and Day") +
  theme_bw()

#Accidents per age--------------------------------------------------------------------------------------------------------------------

dataset <- read.csv('Artif.csv')

dataset$age_band_of_casualty <- gsub('-1', '', dataset$age_band_of_casualty) #Changing -1 to NA
dataset$age_band_of_casualty <- as.numeric(dataset$age_band_of_casualty) #setting column back to numeric form
dataset <- dataset[!is.na(dataset$age_band_of_casualty),] #Removing NA's
dataset$age_band_of_casualty <- as.character(dataset$age_band_of_casualty) #Setting age band as a character

#Setting age band column levels and changing the values to a more friendly name
dataset$age_band_of_casualty <- factor(dataset$age_band_of_casualty,
                                       levels = c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"),
                                       labels = c("0-5", "6-10", "11-15", "16-20", "21-25", "26-35", "36-45",
                                                  "46-55", "56-65", "66-75", "Over 75"))


daily_data3 <- mutate(dataset,Band= age_band_of_casualty,hour=substring(time,12,13)) %>% 
  arrange(desc(Band)) %>%
  group_by(Band, hour) %>% summarize(num_accs=n()) %>% 
  mutate(prop=round(100*num_accs/sum(num_accs), 1)) %>%
  filter(hour!="") 



p <-ggplot(daily_data3, aes(hour, num_accs))
p +geom_bar(stat = "identity", aes(fill = Band ), position = "dodge") +
  xlab("Hour of Day") + 
  ylab("Accident Count") +
  ggtitle("Accidents based on hour and casualty age, 2005-2015") +
  theme_bw()


#Speed by hour------------------------------------------------------------

dataset <- read.csv('Artif.csv')


dataset$speed_limit <- as.character(dataset$speed_limit) #Setting age band as a character

#Setting age band column levels and changing the values to a more friendly name
dataset$speed_limit <- factor(dataset$speed_limit,
                              levels = c("0", "10", "20", "30", "40", "50", "60", "70"),
                              labels = c("stationary", "10 mph", "20 mph", "30 mph", "40 mph",
                                         "50 mph", "60 mph", "70 mph"))



daily_data4 <- mutate(dataset,Speed= speed_limit, hour=substring(time,12,13)) %>% 
  arrange(desc(Speed)) %>%
  group_by(Speed, hour) %>% summarize(num_accs=n()) %>% 
  mutate(prop=round(100*num_accs/sum(num_accs), 1)) %>%
  filter(hour!="") 


#Makes a plot that displays number of accidents based on speed and hour of day.
p <-ggplot(daily_data4, aes(hour, num_accs))
p +geom_bar(stat = "identity", aes(fill = Speed ), position = "dodge") +
  xlab("Hour of Day") + 
  ylab("Accident Count") +
  ggtitle("Accidents based on hour and speed") +
  theme_bw()

#We will drill down further into the dataset to reveal more about accidents at 30mph on single carriageways-----

dataset <- read.csv('Artif.csv')

#Making a seperate column to hold hour of the day
hour_speed <- mutate(dataset, hour=substring(time,12,13))
#Removing the hours 12am to 6am, data is in Char format so < or > would not have worked.
hour_speed <- filter(hour_speed, hour != '00')
hour_speed <- filter(hour_speed, hour != '01')
hour_speed <- filter(hour_speed, hour != '02')
hour_speed <- filter(hour_speed, hour != '03')
hour_speed <- filter(hour_speed, hour != '04')
hour_speed <- filter(hour_speed, hour != '05')
hour_speed <- filter(hour_speed, hour != '06')

#Filter out speed limits which aren't 60mph
hour_speed <- filter(hour_speed, speed_limit == 30)



#Make a new column with the road type as a char-------------------------------------------------------------------
roadT <- read.csv("roadtype.csv", stringsAsFactors=F)
#The hour_speed dataset and the roadT dataset have different spelling for road_type
#We will rename the hour_speed dataset to allign it with the roadT
hour_speed <- rename(hour_speed, Road_Type = road_type)

#Making the new column using the numeric code number to give a character name in new column
hour_speed <- hour_speed %>% 
  left_join(roadT, by=c("Road_Type"="code"))
rm(roadT) #Option to remove the road type dataset from global environment

#Filter the dataset down again to contain only single carriageway accidents---------------------------------------
hour_speed <- filter(hour_speed, Road_Type.y == 'Single carriageway')

#We now have a dataset with time from 7am till 11pm, a speed limit of 30mph, only single carriageways as the road type--
#We can now look at the dataset and drill down to the age bands of drivers who will be involved 
#in accidents based on these variables

#Making a new column with the name of the age bands-------------------------------------------------------------------
AgebandT <- read.csv("Driver_age_Band.csv", stringsAsFactors=F)

hour_speed <- hour_speed %>% 
  left_join(AgebandT, by=c("age_band_of_driver"="code"))

rm(AgebandT)#Option to remove the age band dataset from global environment

#We will remove entries where the data is missing
hour_speed <- filter(hour_speed, age_band_of_driver.y != 'Data missing')


hour_speed1 <- mutate(hour_speed,Band= age_band_of_driver.y,hour=substring(hour,1,2)) %>% 
  arrange(desc(Band)) %>%
  group_by(Band, hour) %>% summarize(num_accs=n()) %>% 
  mutate(prop=round(100*num_accs/sum(num_accs), 1)) %>%
  filter(hour!="") 

#Displays a plot showing accidents per hour based on age band
p <-ggplot(hour_speed1, aes(hour, num_accs))
p +geom_bar(stat = "identity", aes(fill = Band ), position = "dodge") +
  xlab("Hour of Day") + 
  ylab("Accident Count") +
  ggtitle("Accidents based on hour and age band") +
  theme_bw()


#Can be seen that most accidents involve people who are older than 26 but younger than 55
#We can drill down further and look at the data
#Copy hour_speed, this will mean any mistakes can be easier fixed as the original dataset is available
hour_speed2 <- hour_speed 

#Filter out age bands to have just 26-35, 36-45, and 46-55 remain

hour_speed2 <- filter(hour_speed2, age_band_of_driver.y != '0 - 5')
hour_speed2 <- filter(hour_speed2, age_band_of_driver.y != '11 - 15')
hour_speed2 <- filter(hour_speed2, age_band_of_driver.y != '16 - 20')
hour_speed2 <- filter(hour_speed2, age_band_of_driver.y != '21 - 25')
hour_speed2 <- filter(hour_speed2, age_band_of_driver.y != '56 - 65')
hour_speed2 <- filter(hour_speed2, age_band_of_driver.y != '6 - 10')
hour_speed2 <- filter(hour_speed2, age_band_of_driver.y != '66 - 75')
hour_speed2 <- filter(hour_speed2, age_band_of_driver.y != 'Over 75')

#Counts the number in each age band-------------------------------------------------------------------------
numbers <- hour_speed2$age_band_of_driver.y
a <- table(numbers)
a


#We can filter out just the 26-35 year old driver age band for further exploration--------------------------
hour_speed2 <- filter(hour_speed2, age_band_of_driver.y == '26 - 35')
hour_speed3 <- hour_speed2 #Copying the dataset incase any mistakes are made

hour_speed3$age_of_driver <- as.character(hour_speed3$age_of_driver) #Setting age of driver as a character

#Setting driver age olumn levels and changing the values to a more friendly name
hour_speed3$age_of_driver <- factor(hour_speed3$age_of_driver,
                              levels = c("26", "27", "28", "29", "30", "31", "32", "33", "34", "35"),
                              labels = c("26 Years", "27 Years", "28 Years", "29 Years", "30 Years",
                                         "31 Years", "32 Years", "33 Years", "34 Years", "35 Years"))

#We can now explorer the age band 26 to 35
hour_speed3 <- mutate(hour_speed3, Age= age_of_driver, hour=substring(hour,1,2)) %>% 
  arrange(desc(Age)) %>%
  group_by(Age, hour) %>% summarize(num_accs=n()) %>% 
  mutate(prop=round(100*num_accs/sum(num_accs), 1)) %>%
  filter(hour!="") 

#Displays a plot showing accidents per hour based on age band
p <-ggplot(hour_speed3, aes(hour, num_accs))
p +geom_bar(stat = "identity", aes(fill = Age ), position = "dodge") +
  xlab("Hour of Day") + 
  ylab("Accident Count") +
  ggtitle("Accidents based age of driver and time (30mph, single carriageway)") +
  theme_bw()



numbers <- hour_speed2$age_of_driver
a <- table(numbers)
a

#Filter the data to contains entried where the driver age is 30 years old

hour_speed4 <- filter(hour_speed2, age_of_driver == '30')

numbers <- hour_speed2$sex_of_driver
a <- table(numbers)
a










#----------------------------------------------------------------------------------------------------------------------



Datafatal <- filter(dataset, accident_severity == 1)

Datafatal$vehicle_manoeuvre <- gsub('-1', '', Datafatal$vehicle_manoeuvre) #Changing -1 to NA
Datafatal$vehicle_manoeuvre <- as.numeric(Datafatal$vehicle_manoeuvre) #setting column back to numeric form
Datafatal$vehicle_manoeuvre[!is.na(Datafatal$vehicle_manoeuvre),] #Removing NA's
Datafatal$vehicle_manoeuvre <- as.character(Datafatal$vehicle_manoeuvre) #Setting age band as a character

Datafatal$vehicle_manoeuvre <- factor(Datafatal$vehicle_manoeuvre,
                              levels = c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
                                         "16", "17", "18"),
                              labels = c("Reversing", "Parked", "Waiting to go", "Slowing or Stopping", "Moving of", "U-Turn",
                                         "Turning Left", "Waiting to turn left", "Turning right", "waiting to turn right",
                                         "Changing to left lane", "Changing to right lane", "Overtaking moving vehicle-Offside",
                                         "Overtaking static vehicle-Offside", "Overtaking-Nearside", "Going ahead left hand bend",
                                         "Going ahead right hand bend", "Going ahead - other"))



daily_data5 <- mutate(Datafatal,Manoeuvre= vehicle_manoeuvre, Driver= substring(age_of_driver,1,2)) %>% 
  arrange(desc(Manoeuvre)) %>%
  group_by(Manoeuvre, Driver) %>% summarize(num_accs=n()) %>% 
  mutate(prop=round(100*num_accs/sum(num_accs), 1)) 


p <-ggplot(daily_data5, aes(Driver, num_accs))
p +geom_bar(stat = "identity", aes(fill = Manoeuvre ), position = "dodge") +
  xlab("Hour of Day") + 
  ylab("Accident Count") +
  ggtitle("Accidents based on hour and speed") +
  theme_bw()


#----------------------------------------------------------------------------------------------------------------------









