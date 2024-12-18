

library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(poissonreg)
library(parsnip)
library(glmnet)
library(rpart)
library(ranger)
library(stacks)
library(dbarts)
library(embed)
library(GGally)
library(discrim)
library(remotes)
library(tensorflow)
library(keras)
library(dbarts)
library(lightgbm)

ggg_train <- vroom("train.csv") %>% 
  select(-id)
ggg_test <-vroom("test.csv")

# ggg_missing <- vroom("trainWithMissingValues.csv")


# imputation --------------------------------------------------------------

my_recipe <- recipe(type ~ . , data=ggg_missing) %>%
  step_impute_mean(bone_length, rotting_flesh, hair_length)

prep <- prep(my_recipe)
baked <- bake(prep, new_data = ggg_missing)

rmse_vec(ggg_train[is.na(ggg_missing)],baked[is.na(ggg_missing)])



# multinomial random forest -----------------------------------------------

my_recipe <- recipe(type~., data = ggg_train) %>% 
  step_dummy(all_nominal_predictors())

prep <- prep(my_recipe)
bake(prep, new_data=ggg_test)

rf_model <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

rf_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(rf_model)

tuning_grid <- grid_regular(mtry(range= c(1,length(ggg_train)-1)),
                            min_n(),
                            levels = 5)

folds <- vfold_cv(ggg_train, v=5, repeats = 1)

CV_results <- rf_wf %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy))

bestTune <- CV_results %>% 
  select_best()

final_wf <-rf_wf  %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = ggg_train)

rf_preds <-final_wf %>% 
  predict(new_data = ggg_test, type = "class")

kaggle_submission <-tibble(id =ggg_test$id,
                      type = rf_preds$.pred_class)

vroom_write(x=kaggle_submission, file="./ggg_rf4.csv", delim=",")


# multinomial naive bayes -------------------------------------------------

my_recipe <- recipe(type~., data = ggg_train) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type))
  

nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("naivebayes")

nb_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(nb_model)

tuning_grid <- grid_regular(Laplace(),
                            smoothness())


folds <- vfold_cv(ggg_train, v = 5, repeats=1)


CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy))

bestTune <- CV_results %>%
  select_best()

final_wf <- nb_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = ggg_train)

nb_preds <-final_wf %>% 
  predict(new_data = ggg_test, type = "class")

kaggle_submission <-tibble(id =ggg_test$id,
                           type = nb_preds$.pred_class)

vroom_write(x=kaggle_submission, file="./ggg_nb18.csv", delim=",")


# multinomial kkn ---------------------------------------------------------

n <- count(ggg_test)

knn_model <- nearest_neighbor(neighbors = sqrt(n)) %>% 
  set_mode("classification") %>% 
  set_engine("kknn")

knn_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(knn_model) %>% 
  fit(data = ggg_train)

knn_preds <- predict(knn_wf, new_data = ggg_test, type = "class")

kaggle_submission <-tibble(id =ggg_test$id,
                           type = knn_preds$.pred_class)

vroom_write(x=kaggle_submission, file="./ggg_knn2.csv", delim=",")


# neural networks ---------------------------------------------------------


nn_recipe <- recipe(type~., data = ggg_train) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_range(all_numeric_predictors(), min = 0, max = 1)

prep <- prep(nn_recipe)
bake(prep, new_data=ggg_train)

nn_model <- mlp(hidden_units = tune(),
                epochs = 50) %>%
  set_engine("keras") %>%
  set_mode("classification")

nn_wf <- workflow() %>% 
  add_recipe(nn_recipe) %>% 
  add_model(nn_model)

folds <- vfold_cv(ggg_train, v = 5, repeats = 1)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1,50)), #what do i put here?
                            levels=5)

tuned_nn <- nn_wf %>%
  tune_grid(resamples = folds,
            grid = nn_tuneGrid,
            metrics = metric_set(accuracy))

tuned_nn %>%
  collect_metrics() %>%
  filter(.metric=="accuracy") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()

bestTune <- tuned_nn %>%
  select_best()

final_wf <- nn_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = ggg_train)

nn_preds <-final_wf %>% 
  predict(new_data = ggg_test, type = "class")

kaggle_submission <-tibble(id =ggg_test$id,
                           type = nn_preds$.pred_class)

vroom_write(x=kaggle_submission, file="./ggg_nn4.csv", delim=",") # this score is really bad


# BART --------------------------------------------------------------------

my_recipe <- recipe(type~., data = ggg_train)

bart_model <- bart(x.train = matrix(0, 0L, 0L), y.train =  ntree = 100) %>%
  set_engine("dbarts") %>% 
  set_mode("classification")

?bart()

bart_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(bart_model)

tuning_grid <- grid_regular(trees(),
                            levels = 5)

folds <- vfold_cv(ggg_train, v = 5, repeats=1)


CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy))

bestTune <- CV_results %>%
  select_best()

final_wf <- nb_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = ggg_train)

nb_preds <-final_wf %>% 
  predict(new_data = ggg_test, type = "class")

kaggle_submission <-tibble(id =ggg_test$id,
                           type = nb_preds$.pred_class)

vroom_write(x=kaggle_submission, file="./ggg_nb10.csv", delim=",")


# lightgbm ----------------------------------------------------------------


my_recipe <- recipe(type~., data = ggg_train) %>% 
  step_dummy(all_nominal_predictors())

boost_model <- boost_tree(tree_depth = tune(),
                   trees = tune(),
                   learn_rate = tune()) %>%
  set_engine("lightgbm") %>% 
  set_mode("classification")

boost_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(boost_model)

tuning_grid <- grid_regular(tree_depth(), trees(), learn_rate(),
                            levels = 5)

folds <- vfold_cv(ggg_train, v = 5, repeats=1)


CV_results <- boost_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy))

bestTune <- CV_results %>%
  select_best()

final_wf <- boost_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = ggg_train)

boost_preds <-final_wf %>% 
  predict(new_data = ggg_test, type = "class")

kaggle_submission <-tibble(id =ggg_test$id,
                           type = boost_preds$.pred_class)

vroom_write(x=kaggle_submission, file="./ggg_boost.csv", delim=",")


