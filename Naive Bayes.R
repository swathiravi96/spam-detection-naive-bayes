##### Classification using Naive Bayes --------------------

## Example: Filtering spam SMS messages ----
## Step 2: Exploring and preparing the data ---- 
#We will transform our data into a representation known as bag-of-words, 
#which ignores word order and simply 
#provides a variable indicating whether the word appears at all.

# read the sms data into the sms data frame
sms_raw <- read.csv("sms_spam.csv", stringsAsFactors = FALSE)

# examine the structure of the sms data
str(sms_raw)

# convert spam/ham to factor.
sms_raw$type <- factor(sms_raw$type)

# examine the type variable more carefully
str(sms_raw$type)
table(sms_raw$type)

# build a corpus using the text mining (tm) package

#The first step in processing text data involves creating a corpus,
#which is a collection of text 
#In order to create a corpus, we'll use the VCorpus() function in the tm package
#use the VectorSource() reader function to create a source object from the existing
#sms_raw$text vector, which can then be supplied to VCorpus()

install.packages('tm')
library(tm)

sms_corpus <- VCorpus(VectorSource(sms_raw$text))

# examine the sms corpus
print(sms_corpus)

#Because the tm corpus is essentially a complex list, we can use list operations to select
#documents in the corpus. To receive a summary of specific messages, we can use the
#inspect() function with list operators.
#the following command willview a summary of the first and second SMS messages in the corpus

inspect(sms_corpus[1:2])

#To view the actual message text, the as.character() function must be applied to the desired messages.

as.character(sms_corpus[[1]])

#To view multiple documents, we'll need to use as.character() on several items in
#the sms_corpus object. To do so, we'll use the lapply() function.

lapply(sms_corpus[1:2], as.character)

# clean up the corpus using tm_map()

#the corpus contains the raw text of 5,559 text messages. In order
#to perform our analysis, we need to divide these messages into individual words.
#we need to clean the text, in order to standardize the words, by removing
#punctuation and other characters that clutter the result.
#The tm_map() function provides a method 
#to apply a transformation (also known as mapping) to a tm corpus.

#Our first order of business will be to standardize the messages to use only lowercase characters. 
#R provides a tolower() function that returns a lowercase version of text strings. 
#In order to apply this function to the corpus, we need to
#use the tm wrapper function content_transformer() to treat tolower() as a
#transformation function that can be used to access the corpus.

sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))

# show the difference between sms_corpus and corpus_clean

#To check whether the command worked as advertised, let's inspect the first message
#in the original corpus and compare it to the same in the transformed corpus:
as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]])

#Let's continue our cleanup by removing numbers from the SMS messages.

#the majority would likely be unique to individual senders 
#and thus will not provide useful patterns across all messages

sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers) # remove numbers

#remove filler words such as to, and, but, and or from our SMS messages. 
#These terms are known as stop words and are typically removed prior to text mining.
#we'll use the stopwords() function provided by the tm package. 
#This function allows us to access various sets of stop words, across several languages. 
#By default, common English language stop words are used. 
#To see the default list, type stopwords() at the command line.

sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords()) # remove stop words

   #we can also eliminate any punctuation from the text messages using the built-in removePunctuation() 
#transformation

sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation) # remove punctuation

# tip: create a custom function to replace (rather than remove) punctuation
removePunctuation("hello...world")
replacePunctuation <- function(x) { gsub("[[:punct:]]+", " ", x) }
replacePunctuation("hello...world")

# illustration of word stemming
#Another common standardization for text data involves reducing words to their root
#form in a process called stemming. 
#The stemming process takes words like learned, learning, and learns, 
#and strips the suffix in order to transform them into the base form, learn. 
#This allows machine learning algorithms to treat the related terms as a
#single concept rather than attempting to learn a pattern for each variant.

install.packages("SnowballC")
library(SnowballC)
wordStem(c("learn", "learned", "learning", "learns"))

#to apply the wordStem() function to an entire corpus of text documents, the
#tm package includes a stemDocument() transformation.

sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)

#The final step in our text cleanup process is to remove additional whitespace,
#using the built-in stripWhitespace() transformation:

sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace) # eliminate unneeded whitespace

# examine the final clean corpus
lapply(sms_corpus[1:3], as.character)
lapply(sms_corpus_clean[1:3], as.character)

#the final step is to split the messages into individual components through a
#process called tokenization. 
#A token is a single element of a text string; in this case, the tokens are words.

#The DocumentTermMatrix() function will take a corpus and create
#a data structure called a Document Term Matrix (DTM) in which rows indicate
#documents (SMS messages) and columns indicate terms (words).
#Each cell in the matrix stores a number indicating a count of the times the word
#represented by the column appears in the document represented by the row.


# create a document-term sparse matrix directly from the SMS corpus
sms_dtm <- DocumentTermMatrix(sms_corpus, control = list(
  tolower = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  removePunctuation = TRUE,
  stemming = TRUE
))


# creating training and test datasets
#testing. 
#Since the SMS messages are sorted in a random order, we can simply take the
#first 4,169 for training and leave the remaining 1,390 for testing. 
#the DTM object acts very much like a data frame and can be split using 
#the standard [row,col] operations. As our DTM stores SMS messages as rows and words as columns,
#we must request a specific range of rows and all columns for each:
sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test  <- sms_dtm[4170:5559, ]

# also save the labels

#For convenience, it is also helpful to save a pair of vectors with labels for
#each of the rows in the training and testing matrices. 
#These labels are not stored in the DTM, 
#so we would need to pull them from the original sms_raw data frame

sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels  <- sms_raw[4170:5559, ]$type

# check that the proportion of spam is similar

#Both the training data and test data contain about 13 percent spam.
#This suggests that the spam messages were divided evenly between the two datasets.

prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))

# word cloud visualization

#A word cloud is a way to visually depict the frequency at which words appear in
#text data. The cloud is composed of words scattered somewhat randomly around the
#figure. Words appearing more often in the text are shown in a larger font, while less
#common terms are shown in smaller fonts.
#comparing the clouds for spam and ham will help us gauge whether our Naive Bayes spam filter
#is likely to be successful.

#This will create a word cloud from our prepared SMS corpus. 
#Since we specified random.order = FALSE, the cloud will be arranged in a nonrandom order with higher
#frequency words placed closer to the center. 
#The min.freq parameter specifies the number of times a word must appear in the corpus before it will be displayed in the
#cloud. 
#Since a frequency of 50 is about 1 percent of the corpus, this means that a word
#must be found in at least 1 percent of the SMS messages to be included in the cloud.
install.packages("wordcloud")
library(wordcloud)
wordcloud(sms_corpus_clean, min.freq = 50, random.order = FALSE)

# subset the training data into spam and ham groups

#comparing the clouds for SMS spam and ham. 
#Since we did not construct separate corpora for spam and ham,
#this is an appropriate time to note a very helpful feature of the wordcloud()
#function.
spam <- subset(sms_raw, type == "spam")
ham  <- subset(sms_raw, type == "ham")

#We now have two data frames, spam and ham, each with a text feature containing
#the raw text strings for SMSes.
#This time, we'll use the max.words parameter to look at the 40 most common words in
#each of the two sets. 
#The scale parameter allows us to adjust the maximum and
#minimum font size for words in the cloud.
#Because of the randomization process, each word cloud may
#look slightly different.
#Running the wordcloud() function
#several times allows you to choose the cloud that is the most
#visually appealing for presentation purposes.

png('file.jpg')
wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
dev.off()
#Spam messages include words such as urgent, free, mobile, claim, and stop;
#these terms do not appear in the ham cloud at all. 
#Instead, ham messages use words such as can, sorry, need, and time.
#These stark differences suggest that our Naive Bayes model will have some strong
#key words to differentiate between the classes.

png('file2.jpg')
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))
dev.off()

#final step in the data preparation process is to transform the sparse matrix into a
#data structure that can be used to train a Naive Bayes classifier.
#To reduce the number of features, we will eliminate any word that appear in less than five
#SMS messages, or in less than about 0.1 percent of the records in the training data.

sms_dtm_freq_train <- removeSparseTerms(sms_dtm_train, 0.999)
sms_dtm_freq_train

# indicator features for frequent words

#The findFreqTerms() function takes a DTM and returns a character vector containing
#the words that appear for at least the specified number of times. For instance,
#the following command will display the words appearing at least five times in
#the sms_dtm_train matrix

findFreqTerms(sms_dtm_train, 5)

# save frequently-appearing terms to a character vector
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)
#the vector shows us that there are 1,136 terms appearing in at least five SMS messages
str(sms_freq_words)

# create DTMs with only the frequent terms

#We now need to filter our DTM to include only the terms appearing in a specified vector.
#Since we want all the rows, but only the columns representing the words in
#the sms_freq_words vector, our commands:

sms_dtm_freq_train <- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]

# convert counts to a factor

#The Naive Bayes classifier is typically trained on data with categorical features.
#This poses a problem, since the cells in the sparse matrix are numeric and measure
#the number of times a word appears in a message. We need to change this to a
#categorical variable that simply indicates yes or no depending on whether the
#word appears at all.

#ifelse(x > 0, "Yes", "No") statement transforms the values in x, 
#so that if the value is greater than 0, then it will be replaced by "Yes",
#otherwise it will be replaced by a "No" string.

convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

# apply() convert_counts() to columns of train/test data 

#We now need to apply convert_counts() to each of the columns in our sparse matrix
#MARGIN parameter to specify either rows or columns. Here,
#we'll use MARGIN = 2, since we're interested in the columns (MARGIN = 1 is used
                                                             #for rows)
#The result will be two character type matrixes, each with cells indicating "Yes" or
#"No"
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test  <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)

## Step 3: Training a model on the data ----
library(e1071)
#The sms_classifier object now contains a naiveBayes classifier object that can be
#used to make predictions.
sms_classifier <- naiveBayes(sms_train, sms_train_labels)

## Step 4: Evaluating model performance ----

#To evaluate the SMS classifier, we need to test its predictions on unseen messages
##named sms_test, while the class labels (spam or ham) are stored in a vector named
#sms_test_labels. The classifier that we trained has been named sms_classifier.
#We will use this classifier to generate predictions and then compare the predicted
#values to the true values. 

sms_test_pred <- predict(sms_classifier, sms_test)

#add some additional parameters to eliminate unnecessary cell proportions 
#and use the dnn parameter (dimension names) to relabel the rows and columns

library(gmodels)
CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## Step 5: Improving model performance ----
sms_classifier2 <- naiveBayes(sms_train, sms_train_labels, laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)
CrossTable(sms_test_pred2, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
