##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)


# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

########################

str(edx) #structure of the dataset

head(edx) #first entries of the edx dataset

dim(edx) #to know how many rows and columns are in the edx dataset

n_distinct(edx$userId) #how many distinct users the dataset has

edx %>% group_by(userId) %>%  ## Graph of a distribution of users by number of ratings given
  summarise(n=n()) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "blue") +
  ggtitle("Distribution of Users by Number of Ratings given") +
  xlab("Ratings") +
  ylab("Users")  +
  scale_x_log10()

n_distinct(edx$movieId) # how many distinct movies 


edx %>% group_by(movieId) %>% #Graph of a distribution of Movies by number of Ratings
  summarise(n=n()) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "red") +
  ggtitle("Distribution of Movies by number of Ratings given") +
  xlab("Number of Ratings") +
  ylab("Number of Movies") +
  scale_x_log10()




ratings_distribution <- edx %>% group_by(rating) %>% summarise(ratings_d_sum = n()) ## Distribution of Ratings
ratings_distribution



ratings_distribution %>% mutate(rating = factor(rating)) %>% #Graph of Rating Distribution
  ggplot(aes(rating, ratings_d_sum)) +
  geom_col(fill = "grey", color = "white") +
  theme_classic() + 
  labs(x = "Rating", y = "Count",
       title = "Number of ratings")


edx %>% group_by(genres) %>% #First 6 distinct combinations of Genres
  summarise(n=n()) %>% head()

n_distinct(edx$genres) ## how many distinct genres are in the dataset





# Define Root Mean Squared Error (RMSE)
RMSE <- function(true_ratings, pred_ratings){
  sqrt(mean((true_ratings - pred_ratings)^2))
}



#movie effect

# average of all ratings of the edx dataset
mu <- mean(edx$rating)

# calculate b_i on the training set
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(bi = mean(rating - mu))

# predicted ratings
pred_ratings_bi <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$bi

#movie + user effect

#b_u using the training set 
user_avgs <- edx %>%  
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(bu = mean(rating - mu - bi))

#predicted ratings
pred_ratings_bu <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + bi + bu) %>%
  .$pred


#RMSES for Movies and User effects

rmse1 <- RMSE(validation$rating,pred_ratings_bi)  
rmse1


rmse2 <- RMSE(validation$rating,pred_ratings_bu)
rmse2


# Regularization 

# Lambda is a tuning parameter. We can use cross-validation to choose it.

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu_reg <- mean(edx$rating)
  
  bi_reg <- edx %>% 
    group_by(movieId) %>%
    summarize(bi_reg = sum(rating - mu_reg)/(n()+l))
  
  bu_reg <- edx %>% 
    left_join(bi_reg, by="movieId") %>%
    group_by(userId) %>%
    summarize(bu_reg = sum(rating - bi_reg - mu_reg)/(n()+l))
  
  pred_ratings_biu <- 
    validation %>% 
    left_join(bi_reg, by = "movieId") %>%
    left_join(bu_reg, by = "userId") %>%
    mutate(pred = mu_reg + bi_reg + bu_reg) %>%
    .$pred
  
  return(RMSE(validation$rating,pred_ratings_biu))
})


#Lambdas by Rmses

qplot(lambdas, rmses)  

#Minimum Lambda

lambda <- lambdas[which.min(rmses)]
lambda

#RMSE for Regularization

rmse3 <- min(rmses)
rmse3




