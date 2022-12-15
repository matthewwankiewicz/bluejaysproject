library(tidyverse)
library(caret)
library(visdat)
library(lmtest)
library(pROC)
library(randomForest)
library(ROSE)

## read in and clean data ----------------------------------------------------
training <- read_csv("training.csv")

## change spin rate to numeric
training$SpinRate <- as.numeric(training$SpinRate)


## find missing values
colSums(is.na(training))

## impute missing values with average of column
training <- training %>% 
  mutate(SpinRate = replace_na(SpinRate, mean(training$SpinRate, na.rm = T)))

## explore data ---------------------------------------------------------------

## characteristics of balls in play
training %>% 
  filter(InPlay == 1) %>% 
  summary()

## characteristics of balls not in play
training %>% 
  filter(InPlay == 0) %>% 
  summary()

### it appears that balls in play have slightly lower velocity, spin rates and
### higher Horizontal and Induced Vertical Breaks

vis_cor(training)

cols <- c("Red", "Blue")
pairs(training,
      col = cols[as.factor(training$InPlay)])

training %>% 
  ggplot(aes(InPlay)) +
  geom_bar() +
  labs(title = "Distribution of balls in play vs not in play")

### build models --------------------------------------------------------------

### create training and testing samples
set.seed(111)
sample <- sample(c(TRUE, FALSE), nrow(training), replace=TRUE, prob=c(.7,.3))
train  <- training[sample, ]
test   <- training[!sample, ]


## start with basic model - all four variables, no interactions

model1 <- glm(InPlay ~ ., data = train, family = binomial())

## get summary of the model
summary(model1)


## create estimates on testing set
test$estimate <- predict(model1, newdata = test,
                         type = "response")

test <- test %>% 
  mutate(est_ip = ifelse(estimate > .5, 1, 0))

## create a confusion matrix for logistic model
confusionMatrix(reference = as.factor(test$InPlay), as.factor(test$est_ip), positive = "1")


#### while the model gives us a good idea of the importance of each variable, 
#### it's estimates are very low, meaning we cannot look at its ROC curve, AUC and
#### and other performance metrics using a cutoff of 0.5


## create model including interactions between variables
model2 <- glm(InPlay ~ .^2, 
             data = train, family = binomial())

## summary of the model
summary(model2)


## create estimates on testing set
test$estimate <- predict(model2, newdata = test,
                         type = "response")

test <- test %>% 
  mutate(est_ip = ifelse(estimate > .5, 1, 0))

## create a confusion matrix for logisitic model
confusionMatrix(reference = as.factor(test$InPlay), as.factor(test$est_ip), positive = "1")

## ROC plot and AUC
roc(test$est_ip, test$InPlay, plot = T)


### this model is much more difficult to interpret than our initial model but
### it has very strong results (very high AUC). We can attempt to improve
### upon this model using step wise selection to find a better overall model

## step wise selection (AIC)
model_step <- step(model2, direction = "both")

## summary of new model
summary(model_step)


## create estimates on testing set
test$estimate <- predict(model_step, newdata = test,
                             type = "response")

test <- test %>% 
  mutate(est_ip = ifelse(estimate > .5, 1, 0))

## create a confusion matrix for logisitic model
confusionMatrix(reference = as.factor(test$InPlay), as.factor(test$est_ip), positive = "1")

## ROC plot and AUC
roc(test$est_ip, test$InPlay, plot = T)

precision(test, as.factor(InPlay), as.factor(est_ip))

### our new model returns identical results to the more complex model. We will
### use the newer model as it is easier to interpret the impact of each variable
### on the chance a ball is put in play


## compare new model to null model and initial model

null_mod <- glm(InPlay ~ 1, data = train, family = binomial())

lrtest(null_mod, model_step)

lrtest(model1, model_step)


### both likelihood ratio tests yield p-values below 0.05, meaning that the final
### model we created has the best fit of the data so far

lrtest(model2, model_step)

### After testing the complex and step wise models, we cannot conclude that there
### is a significant difference. Therefore, we will use the simpler one.

### random forest - determine if random forest yields better results compared to
### the logistic regression model


rf <- randomForest(as.factor(InPlay) ~ ., data = train)

print(rf)

preds <- predict(rf, newdata = test)

test$est_ip <- preds

confusionMatrix(reference = as.factor(test$InPlay), as.factor(test$est_ip), positive = "1")

## ROC plot and AUC
roc(test$est_ip, test$InPlay, plot = T)

### these results are not as strong as the logistic regression model, try to find optimal
### mtry value before settling on logistic
set.seed(111)
tuneRF(train[-1], as.factor(train$InPlay))


### optimal value is 1, lets try again
rf2 <- randomForest(as.factor(InPlay) ~ ., data = train, mtry = 1)

print(rf2)

preds <- predict(rf2, newdata = test)

test$est_ip <- preds

confusionMatrix(reference = as.factor(test$InPlay), as.factor(test$est_ip), positive = "1")

## ROC plot and AUC
roc(test$est_ip, test$InPlay, plot = T)

## find important variables
varImpPlot(rf2)

### results from the logistic regression are still the best and are also the easiest
### to interpret, will now try re-sampling the response to see if that changes results

## re-sample ------------------------------------------------------------
set.seed(111)
over_sampled <- ovun.sample(InPlay ~ ., data = train, method = "both",
                            N = 6971)$data


## re-attempt fitting logistic regression and random forest models


### logistic

model_samp <- glm(InPlay ~ ., data = over_sampled, family = binomial())

## summary of new model
summary(model_samp)


## create estimates on testing set
test$estimate <- predict(model_samp, newdata = test,
                         type = "response")

test <- test %>% 
  mutate(est_ip = ifelse(estimate > .5, 1, 0))

## create a confusion matrix for logisitic model
confusionMatrix(reference = as.factor(test$InPlay), as.factor(test$est_ip), positive = "1")

## ROC plot and AUC
roc(test$est_ip, test$InPlay, plot = T)

precision(test, as.factor(InPlay), as.factor(est_ip))

### while the AUC is much lower, this model allows us to have easy to interpret 
### coefficients and is able to detect negative occurrences at a higher rate
### this model will likely be better to use compared to the step wise model

### random forest
rf_samp <- randomForest(as.factor(InPlay) ~ ., data = over_sampled, mtry=1)

print(rf_samp)

preds <- predict(rf_samp, newdata = test)

test$est_ip <- preds

confusionMatrix(reference = as.factor(test$InPlay), as.factor(test$est_ip), positive = "1")

## ROC plot and AUC
roc(test$est_ip, test$InPlay, plot = T)

## find important variables
varImpPlot(rf_samp)

### we still have better results from our logistic regression model, 
### this will be the model we use to create predictions

final_model <- model_samp

write_rds(final_model, "final_model.rds")

## display coefficients of final model

sjPlot::plot_model(model_samp) +
  labs(subtitle = "Variables less than 1 decrease contact probability, greater than 1 increases") +
  scale_y_continuous(limits = c(0.9, 1.1))


### prep "deploy" files -------------------------------------------------------

deploy <- read_csv("deploy.csv")
deploy$SpinRate <- as.numeric(deploy$SpinRate)


## find missing values
colSums(is.na(deploy))

## impute missing values with average of column
deploy <- deploy %>% 
  mutate(SpinRate = replace_na(SpinRate, mean(deploy$SpinRate, na.rm = T)))


write_csv(deploy, "deploy.csv")
