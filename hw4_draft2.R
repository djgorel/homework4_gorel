#install.packages("modelr")
#install.packages("psych")
library(caret)
library(naivebayes)
library(ISLR)
library(dplyr)
library(modelr)
library(psych)


#enter your path here
loan_data <- read.csv("/home/devon/Documents/loan_approval_dataset.csv")
loan_data <- na.omit(loan_data) 

#create train test split

Ind <- sample(2,nrow(loan_data),replace=T,prob=c(0.8,0.2))#creates the train test split
train <- loan_data[Ind==1,]
test <- loan_data[Ind==2,]
#creates model
#predict grade using the age, income, home ownershi[, length of e,ployment, and if they have defaulted]
#bayes_model <- naive_bayes(risk~ person_age + person_income + person_home_ownership + person_emp_length + loan_intent + loan_amnt, data=train, laplace=1,usekernel=T)
bayes_model <- naive_bayes(loan_status~ no_of_dependents + education + self_employed + income_annum + loan_amount + loan_term + cibil_score + residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value, data=train, laplace=1,usekernel=T)

plot(bayes_model)
p <- predict(bayes_model,test)
t_bayes <- table(p, test$loan_status)
t_bayes
#to print the accuracy of the classifier
print(paste("Accuracy of naive bayes is", sum(diag(t_bayes))/sum(t_bayes)))

#this is a modelr function, should add the prediction to the dataframe
test_bayes <- add_predictions(test, bayes_model, var = "Prediction")

#test
#trying to plot it
#either of these work
ggplot(test_bayes, aes(x=loan_status,fill=Prediction)) + geom_bar(stat='count',position='dodge') + labs(title="Naive Bayes Loan Approval Classification", x= "Loan Approval Status", y= "Count")


#for logistical regression we will make a new model
glm.fit <- glm(loan_status~ no_of_dependents + education + self_employed + income_annum + loan_amount + loan_term + cibil_score + residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value, data=train, family=binomial)
glm.probs <- predict(glm.fit,test,type="response")

glm.pred <- ifelse(glm.probs>0.5,"Rejected","Approved")

attach(test)

t_log <- table(glm.pred,loan_status)
t_log

print(paste("Accuracy of logistic regression is", sum(diag(t_log))/sum(t_log)))



#adding the prediction of the logistic to a appended test dataset

test_logistic <- mutate(test,Prediction=glm.pred)

ggplot(test_logistic, aes(x=loan_status,fill=Prediction)) + geom_bar(stat='count',position='dodge') + labs(title="Logistic Regression Loan Approval Classification", x= "Loan Approval Status", y= "Count")
