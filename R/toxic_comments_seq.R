## load libraries
library(here)
library(tidyverse)
library(keras)

vocab_size = 10000
max_len = 200

## load data
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
  layer_embedding(input_dim = vocab_size, output_dim = 16) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 6, activation = "sigmoid")

## specify model optimizer, loss and metrics
model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)

## fit
history <- model %>% fit(
  x_train,
  y_train,
  epochs = 16,
  batch_size = 32,
  validation_split = 0.2,
  verbose=1
  # ,callbacks = c(early_stopping)
)