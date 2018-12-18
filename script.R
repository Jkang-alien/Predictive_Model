library(AmesHousing)
data("ames_raw")
library(purrr)
map(ames_raw, names)

names(ames_raw) <- gsub(' ', '_', names(ames_raw))
map(ames_raw, class)

library(dplyr)
ames_raw <- ames_raw %>% 
    mutate(log_sale_price = log10(SalePrice))
summary(lm(log_sale_price ~ Lot_Area, ames_raw))

library(tidyverse)
library(rsample)

################################################################
############ DATA SPLITING #####################################

data_split <- initial_split(ames_raw, strata = 'log_sale_price')
dim(training(data_split))
dim(testing(data_split))

summary(lm(log_sale_price ~ Lot_Area, training(data_split)))


################################################################
#################### Model accessment ##########################

cv_splits <- vfold_cv(training(data_split))

library(yardstick)
lm_fit <- function(data_split, ...)
  lm(..., data = analysis(data_split))

# A formula is also needed for each model:

form <- as.formula(
  log10(Sale_Price) ~ Longitude + Latitude
)

model_perf <- function(data_split, mod_obj) {
  vars <- rsample::form_pred(mod_obj$terms)
  assess_dat <- assessment(data_split) %>%
    select(!!!vars, Sale_Price) %>%
    mutate(
      pred = predict(
        mod_obj,
        newdata = assessment(data_split)
      ),
      Sale_Price = log10(Sale_Price)
    )
  rmse <- assess_dat %>%
    rmse(truth = Sale_Price, estimate = pred)
  rsq <- assess_dat %>%
    rsq(truth = Sale_Price, estimate = pred)
  data.frame(rmse = rmse, rsq = rsq)
}



map(cv_splits$splits, function(x){lm(log_sale_price ~ ., assessment(x))})

assessment(cv_splits$splits[[1]])


#####################################################################
##################### Caret #########################################

library(caret)
load("car_data.RData")

summary(car_data)
hist(car_data$mpg)

qplot(car_data$mpg, bins = 30)
qplot(car_data$model_year)

library(dplyr)
test_set <- car_data %>%
  filter (model_year == 2018)
train_set <- car_data %>%
  filter (model_year < 2018)

library(recipes)

basic_rec <- recipe(mpg ~ ., data = train_set) %>%
  update_role(carline, new_role = "car name") %>%
  step_other(division, threshold = 0.005) %>%
  step_dummy(all_nominal(), -carline) %>%
  step_zv(all_predictors())

glmn_grid <- expand.grid(alpha = seq(0, 1, by = .25), lambda = 10^seq(-3, -1, length = 20))
nrow(glmn_grid)
glmn_grid
car_data$carline[1:10]

ctrl <- trainControl(
  method = 'cv',
  savePredictions = 'final',
  verboseIter = TRUE)

glmn_rec <- basic_rec %>%
  step_poly(eng_displ) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())

set.seed(3544)
glmn_mod <- train(
  glmn_rec,
  data = train_set,
  method = "glmnet",
  trControl = ctrl,
  tuneGrid = glmn_grid
)


ggplot(glmn_mod) + scale_x_log10() + theme(legend.position = 'top')

ggplot(glmn_mod$pred, aes(x = obs, y = pred)) +
  geom_abline(col = 'green', alpha = .5) +
  geom_point(alpha = 0.3) +
  geom_smooth(se = FALSE, col = 'red', lty = 2, lwd = 1, alpha = 0.5)

ggplot(glmn_mod$pred, aes(x = obs, y = pred)) +
  geom_abline() + 
  geom_point()
reg_imp <- varImp(glmn_mod, scale = FALSE)
ggplot(reg_imp, top = 30) + xlab(''  )

library(glmnet)
plot(glmn_mod$finalModel, xvar = 'lambda')
glmn_mod$finalModel

rec_trained <- prep(glmn_rec, training = train_set, retain = TRUE)
baked_data <- bake(rec_trained, new_data = train_set, all_predictors())

######################################################
################# MARS ###############################

ctrl$verboseIter <- FALSE

mars_grid <- expand.grid(degree = 1:2, nprune = seq(2, 60, by = 2))

set.seed(3544)
mars_mod <- train(
  basic_rec, 
  data = train_set,
  method = "earth",
  tuneGrid = mars_grid,
  trControl = ctrl
)

test_set <- test_set %>%
  mutate(pred = predict(glmn_mod, test_set))

library(yardstick)
rmse(test_set, truth = mpg, estimate = pred)

ggplot(test_set, aes(x = mpg, y = pred)) + 
  geom_abline() +
  geom_point() +
  geom_smooth()

set.seed(3544)

mars_gcv_bag <- train(
  basic_rec,
  data = train_set,
  method = "bagEarthGCV",
  tuneGrid = data.frame(degree = 1:2),
  trControl = ctrl,
  B = 50
)

mars_gcv_bag

rs <- resamples (
  list(glmn = glmn_mod, MARS = mars_mod, bagged = mars_gcv_bag)
)

library(tidyposterior)



###################################################################
############# Classification ######################################

library(yardstick)
library(dplyr)


two_class_example %>% head(4)

two_class_example %>% conf_mat(truth = truth, estimate = predicted)

two_class_example %>% accuracy(truth = truth, estimate =  predicted)
library(pROC)

roc_obj <- roc(
  response = two_class_example$truth,
  predictor = two_class_example$Class1,
  levels = rev(levels(two_class_example$truth))
)

auc(roc_obj)

plot(
  roc_obj,
  legacy.axes = TRUE,
  print.thres = c(.2, .5, .8),
  print.thres.pattern = "cut = %.2f (Spec = %.2f, Sens = %.2f)",
  print.thres.cex = .8
)

table(okc_train$Class)

library(caret)
ctrl <- trainControl(
  method = 'cv',
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = 'final',
  sampling = 'down'
)

set.seed(5515)

cart_mod <- train(
  x = okc_train[, names(okc_train) != "Class"],
  y = okc_train$Class,
  method = 'rpart2',
  metric = "ROC",
  tuneGrid = data.frame(maxdepth = 1:20),
  trControl = ctrl
)

cart_mod$finalModel

plot_roc <- function(x, ...) {
  roc_obj <- roc(
    response = x[["obs"]],
    predictor = x[["stem"]],
    levels = rev(levels(x$obs))
  )
  plot(roc_obj, ...)
}

plot_roc(cart_mod$pred)
cart_mod$pred

confusionMatrix(cart_mod)

car_imp <- varImp(cart_mod)
ggplot(car_imp, top = 7)

set.seed(5515)
cart_bag <- train(
  x = okc_train[, names(okc_train) != "Class"],
  y = okc_train$Class,
  methods = 'treebag',
  metric = 'ROC',
  trControl = ctrl
)

cart_bag

confusionMatrix(cart_bag)

plot_roc(cart_mod$pred)
plot_roc(cart_bag$pred,
         col = 'darkred',
         add = TRUE)

bag_imp <- varImp(cart_bag, scale = FALSE)
ggplot(bag_imp, top = 30)


library(recipes)
is_dummy <- vapply(okc_train, function(x) length(unique(x)) == 2 & is.numeric(x), logical(1))
dummies <- names(is_dummy)[is_dummy]
no_dummies <- recipe(Class ~ ., data = okc_train) %>%
  step_bin2factor(!!! dummies) %>%
  step_zv(all_predictors())
smoothing_grid <- expand.grid(usekernel = TRUE, fL = 0, adjust = seq(0.5, 3.5, by = 0.5))

set.seed(5515)
nb_mod <- train(
  no_dummies,
  data = okc_train,
  methods = 'nb',
  metric = 'roc',
  tuneGrid = smoothing_grid,
  trControl = ctrl
)
