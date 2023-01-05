setwd("D:\\learning\\O3\\randomforest")

library(plyr)
# library(ranger)
library(randomForest)

pop<-read.csv("test\\res2nolucpop.csv",encoding="UTF-8")
pop[pop=='-9999'] <- NA
my_pop<-na.omit(pop[,c(1:50)])


#Divide into ten random arrays
CVgroup <- function(k, datasize, seed) {
  cvlist <- list()
  set.seed(seed)
  n <- rep(1:k, ceiling(datasize/k))[1:datasize] #Divide the data into K parts and generate a complete dataset n
  temp <- sample(n, datasize)  #shuffle n
  x <- 1:k
  dataseq <- 1:datasize 
  cvlist <- lapply(x, function(x) dataseq[temp==x])  #Ten random ordered data columns are randomly generated in dataseq
  return(cvlist)
}

k<-10
datasize<-nrow(my_pop)
cvlist <- CVgroup(k = k, datasize = datasize, seed = 130)

mydata<-my_pop

RMSE=rep(0,10)
MAE=rep(0,10)#Empty array, put results in
R2=rep(0,10)
set.seed(904)


for(i in 1:10){
  m=cvlist[[i]]#Group i in the separated cvlist is used as the validation set
  model <- randomForest(station~.,data = mydata[-m,],ntree = 500)
  mypre<-predict(model,mydata[m,])#Put x in the ith group into the model generated with the remaining 9 groups of data, and produce a result
  RMSE[i]<-sqrt(mean((mypre-mydata[m,c('station')])^2))
  MAE[i]<-mean(abs(mypre-mydata[m,c('station')]))
  R2[i]<- 1-(sum((mypre-mydata[m,c('station')])^2)/sum((mydata[m,c('station')]-mean(mydata[m,c('station')]))^2))
}


model$importance
mean(RMSE)
print(RMSE)
mean(MAE)
print(MAE)
mean(R2)
print(R2)
# write.table(mypre, file ="gbbmethodresult.csv", sep = ",", col.names = NA,qmethod = "double")
# write.table(mypre1, file ="rfresult.csv", sep = ",", col.names = NA,qmethod = "double")
# write.table(mydata[m,3], file ="truevalue.csv", sep = ",", col.names = NA,qmethod = "double")

