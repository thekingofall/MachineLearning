---
title: "KNN"
author: "MLP"
date: "2021/1/6"
output: html_document
---
```r
######## 第8章: K-近邻和朴素贝叶斯 ########


## 统一设置ggplot2的绘图风格
library(ggplot2)
theme_set(theme_bw())
```


```r
#install.packages("psych")


#### 8.1：KNN算法 ####

### 对主成分分析降维后的图像数据集进行KNN分类


## 主成分分析对数据进行降维
library(R.matlab)
library(psych)
ETHdata <- readMat("/home/u1935031003/Rstudy/Rstat/Windows/data/chap6/ETH_8class_object_8_big_classes_32_32_1024D.mat")

ETHims <- t(ETHdata$A / 255.0)
dim(ETHims)
labels <- as.vector(ETHdata$labels)
table(labels)
## 可视化碎石图，选择合适的主成分数
parpca <- fa.parallel(ETHims,fa = "pc")
## 可视化碎石图的部分图像
pcanum <- 50
plotdata <- data.frame(x = 1:pcanum,pc.values = parpca$pc.values[1:pcanum])
ggplot(plotdata,aes(x = x,y = pc.values))+
  theme_bw()+
  geom_point(colour = "red")+geom_line(colour = "blue")+
  labs(x = "主成分个数")
```


```r
## 提取前40个主成分
ETHcor <- cor(ETHims,ETHims)
dim(ETHcor)
ETHpca2 <- principal(ETHims,nfactors = 40)
## 使用pca模型获取数据集的40个主成分
ETHpca40<- predict.psych(ETHpca2,ETHims)



### 使用这40个主成分建立KNN模型

#install.packages("caret")
library(caret)
library(Metrics)
library(dplyr)
## 使用这40个主成分建立KNN模型

## 切分训练集和测试集
set.seed(12)
index <- createDataPartition(labels,p=0.75)
labels <- as.factor(labels)
train_ETH <- ETHpca40[index$Resample1,]
train_lab <- labels[index$Resample1]
test_ETH <- ETHpca40[-index$Resample1,]
test_lab <- labels[-index$Resample1]

## KNN分类器
ETHknn <- knn3(x=train_ETH,y=train_lab,k=5)
ETHknn
test_pre <- predict(ETHknn,test_ETH,type = "class")
## 计算KNN模型的精度
table(test_lab,test_pre)
sprintf("KNN分类精度为%.4f",accuracy(test_lab,test_pre))
## 使用混淆矩阵热力图可视化哪些预测正确
confum <- confusionMatrix(test_lab,test_pre)
confum
confumat <- as.data.frame(confum$table)
confumat[,1:2] <- apply(confumat[,1:2],2,as.integer)

ggplot(confumat,aes(x=Reference,y = Prediction))+
  geom_tile(aes(fill = Freq))+
  geom_text(aes(label = Freq))+
  scale_x_continuous(breaks = c(0:8))+
  scale_y_continuous(breaks = unique(confumat$Prediction),
                     trans = "reverse")+
  scale_fill_gradient2(low="darkblue", high="lightgreen", 
                       guide="colorbar")+
  ggtitle("KNN分类器在测试集结果")

## 参数搜索，找到精度更高的模型
set.seed(123)
## 使用交叉验证,5 fold cv
trcl <- trainControl(method="cv",number = 5) 
trgrid <- expand.grid(k=seq(1,25,2))
ETHknnFit <- train(x=train_ETH,y=train_lab, method = "knn",
                   trControl = trcl,tuneGrid = trgrid)


ETHknnFit
## plot 近邻数和精度的关系
plot(ETHknnFit,main="KNN")
```


```r
### KNN 回归



## 美国不同地区的平均房价预测数据集
## 读取数据
house <- read.csv("/home/u1935031003/Rstudy/Rstat/Windows/data/chap5/USA_Housing.csv")
dim(house)
colnames(house)
## 数据标准化，切分为训练数据和测试数据
set.seed(12)
house[,1:5] <- apply(house[,1:5],2,scale)
index <- createDataPartition(house$AvgPrice,p=0.7)
train_house <- house[index$Resample1,]
test_house <- house[-index$Resample1,]
## KNN 回归
houseknn <- knnreg(AvgPrice~.,train_house,k=5)
housetest_pre <- predict(houseknn,test_house)
errormae <- mae(test_house$AvgPrice,housetest_pre)
sprintf("KNN回归的绝对值误差为%.2f",errormae)

## 分析不同的K值下，房价对测试集的预测误差
ks <- seq(1,30,2)
pricemae <- ks
for(ii in 1:length(ks)){
  houseknnii <- knnreg(AvgPrice~.,train_house,k=ks[ii])
  housetest_preii <- predict(houseknnii,test_house)
  pricemae[ii] <- mae(test_house$AvgPrice,housetest_preii)
}
data.frame(k = ks,error_mae = pricemae)%>%
  ggplot(aes(x=k,y=error_mae))+
  geom_line()+geom_point(colour="red")+
  scale_x_continuous(breaks = ks)+
  labs(x="近邻数量",y = "绝对值误差",title = "房价的KNN回归")
```


```r
#### 8.2：朴素贝叶斯方法 ####

### 针对垃圾邮件数据，进行识别


#install.packages("wordcloud")
library(tm)
library(dplyr)
library(tidytext)
library(tidyr)
library(pheatmap)
library(textreg)
library(reshape2)
library(wordcloud)
library(stringr)
## 读取数据
spam <- read.csv("/home/u1935031003/Rstudy/Rstat/Windows/data/chap8/spam.csv",
                 stringsAsFactors = F)

## 对没有正确读取的数据进行修正
strjoin <- function(x){
  x <- as.vector(x)
  text <- ""
  for (ii in 1:length(x)){
    text <- str_c(text,x[ii],sep = " ")
  }
  return(text)
}
spam[,2] <- apply(spam[,2:ncol(spam)],1, strjoin)
spam[,3:ncol(spam)] <- NULL
colnames(spam) <- c("label","text")
spam$label <- as.factor(spam$label)
table(spam$label)

##构建语料库,
spam_cp <- Corpus(VectorSource(spam$text))
## 剔除非英文字符
deletnoneEng <- function(s){
  gsub(pattern = '[^a-zA-Z0-9\\s]+',
       x = s,replacement = "",
       ignore.case = TRUE,
       perl = TRUE)
  }
spam_cp <- tm_map(spam_cp, content_transformer(deletnoneEng))
## 去处语料库中的所有数字
spam_cp <- tm_map(spam_cp,removeNumbers)
## 从文本文档中删除标点符号
spam_cp <- tm_map(spam_cp,removePunctuation)
## 将所有的字母均转化为小写
spam_cp_clearn <- tm_map(spam_cp,tolower)
## 去除停用词
spam_cp <- tm_map(spam_cp,removeWords,stopwords())
## 去除额外的空格
spam_cp <- tm_map(spam_cp,stripWhitespace)
## 将文本词干化
spam_cp <- tm_map(spam_cp,stemDocument)
## 可视化两种情感文本的词云
spam_pro <- data.frame(text=sapply(spam_cp, identity), stringsAsFactors=F)
spam_pro$label <- spam$label
wordfre <- spam_pro%>%unnest_tokens(word,text)%>%
  group_by(label,word)%>%
  summarise(Fre = n())%>%
  arrange(desc(Fre)) %>%
  acast(word~label,value.var = "Fre",fill = 0)
  
## 可视化两种类型邮件的词云
comparison.cloud(wordfre,scale=c(4,.5),max.words=180,
                 title.size=1.5,colors = c("gray50","gray10"))
```


```r
## 找到频繁出现的词语,出现频率大于5
dict <- names(which(wordfre[,1]+wordfre[,2] >5))

## 构建TF矩阵
spam_dtm <- DocumentTermMatrix(spam_cp,control = list(dictionary = dict))
spam_dtm
## 通常情况下，文档－词语的tf-idf矩阵非常的稀疏，可以删除一些不重要的词来缓解矩阵的稀疏性，同时提高计算效率
dim(spam_dtm)                         
spam_dtm <- removeSparseTerms(spam_dtm,0.999)
spam_dtm
## 此时文档－词语的tf-idf矩阵稀疏性得到了缓解，而且词语的数量也减少到了2194个
## 可视化随机抽取100行和100列可视化tf-idf矩阵热力图
set.seed(123)
index <- sample(min(dim(spam_dtm)),100)
pheatmap(as.matrix(spam_dtm)[index,index],cluster_rows = FALSE,
         cluster_cols = FALSE,show_rownames = FALSE, 
         show_colnames = T,main = "TF Matrix Part",
         fontsize_col = 5)
```


```r
save(spam_cp,file = "data/chap8/spam_cp.RData")
save(spam,file = "data/chap8/spam.RData")
```


```r
### 贝叶斯分类器

library(e1071)
library(naivebayes)
library(Metrics)
library(gmodels)
library(ROCR)
## 数据随机切分为75%训练集和25%测试集
set.seed(123)
index <- sample(nrow(spam),nrow(spam)*0.75)
spam_dtm2mat <- as.matrix(spam_dtm)
train_x <- spam_dtm2mat[index,]
train_y <- spam$label[index]
test_x <- spam_dtm2mat[-index,]
test_y <- spam$label[-index]

## 将每个词项所代表的特征转化为对应单词的因子变量
train_x <- apply(train_x, 2, function(x) as.factor(ifelse(x>0,1,0)))
test_x <- apply(test_x, 2, function(x) as.factor(ifelse(x>0,1,0)))

## 使用e1071包中的naiveBayes建立模型
spamnb <- naiveBayes(x = train_x,y = train_y,laplace = 1)
summary(spamnb)
## 对测试集进行预测，查看模型的精度
test_pre <- predict(spamnb,test_x,type = "class")

CrossTable(test_y,test_pre,prop.r = F,prop.t = F,prop.chisq = F)
table(test_y,test_pre)
sprintf("朴素贝叶斯的识别精度为%.4f",accuracy(test_y,test_pre))

pred <- prediction(as.integer(test_pre),as.integer(test_y))
par(pty=c("s"))
plot(performance(pred,"tpr","fpr"),main = "Naive Bayes")




## 使用naivebayes包中的naive_bayes建立模型
## spamnb2 <- naive_bayes(x = train_x,y = train_y,laplace = 1)
## summary(spamnb2)
## 对测试集进行预测，查看模型的精度
## test_pre2 <- predict(spamnb2,test_x,type = "class")
## table(test_pre2,test_y)
## accuracy(test_y,test_pre2)








```

