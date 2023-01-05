setwd("D:\\learning\\O3\\randomforest\\final")

library(plyr)
# library(ranger)
library(randomForest)

pop<-read.csv("res174station.csv",encoding="UTF-8")#Reading modeling data
pop[pop=='-9999'] <- NA
my_pop<-na.omit(pop[,c(1:51)])

daiyuce<-read.csv("res174.csv",encoding="UTF-8")
pop[pop=='-9999'] <- NA
my_yuce<-na.omit(daiyuce[,c(1:50)])#Reading input data

model <- randomForest(station~.,data = my_pop,ntree = 500)
mypre<-predict(model,my_yuce)
write.table(mypre, file ="result174.csv", sep = ",", col.names = NA,qmethod = "double")
# write.table(mypre1, file ="rfresult.csv", sep = ",", col.names = NA,qmethod = "double")
# write.table(mydata[m,3], file ="truevalue.csv", sep = ",", col.names = NA,qmethod = "double")


