## Multi-label classification using data from Kaggle Competition:
## id 8076
## url https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

## load libraries
library(here)
library(dplyr)
library(readr)
library(keras)

## define paramaters
vocab_size = 10000
max_len = 200

## TODO ## ---------------------------------------------------------------------
## Add part to download data and put in /data/toxic_comments/

## load data -------------------------------------------------------------------
train_data <- read_csv(paste0(here::here(), "/data/toxic_comments/train.csv"))
test_data <- read_csv(paste0(here::here(), "/data/toxic_comments/test.csv"))

## use keras tokenizer
tokenizer <- text_tokenizer(num_words = vocab_size) %>% 
  fit_text_tokenizer(train_data$comment_text)

## create sequances
train_seq <- texts_to_sequences(tokenizer, train_data$comment_text)
test_seq <- texts_to_sequences(tokenizer, test_data$comment_text)

## pad sequence
x_train <- pad_sequences(train_seq, maxlen = max_len, padding = "post")
x_test <- pad_sequences(test_seq, maxlen = max_len, padding = "post")

## extract targets columns and convert to matrix
y_train <- train_data %>% 
  select(toxic:identity_hate) %>% 
  as.matrix()


## define model
model <- keras_model_sequential()
model %>% 
  layer_embedding(input_dim = vocab_size, output_dim = 64) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 6, activation = "sigmoid")

## specify model optimizer, loss and metrics
model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)

## fit
history <- model %>% 
  fit(x_train, y_train,
      epochs = 16,
      batch_size = 64,
      validation_split = 0.05,
      verbose = 1)

## predict on test data --------------------------------------------------------
predicted_prob <- predict_proba(model, x_test)

## join ids and predictions -------------------------------------------
res <- as_data_frame(predicted_prob)
names(res) <- names(train_data)[3:8] ## labels names
res <- tibble::add_column(res, id = test_data$id, .before = 1) ## add id column
