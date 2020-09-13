####FOR CHORD SYMBOL RECOGNITION PROJECT#################
rm(list = ls())
### LOAD MODEL IN CURRENT DIR #############
#dir= getwd()
setwd("/Users/tianxuehu/documents/ChordSymbolRec/src/nct_model")
model <- readRDS("final_model.rds")
#####################LOAD .krn MELODY##############################################################################################################
setwd("/Users/tianxuehu/documents/ChordSymbolRec/datasets/haydn_op20_harm/op20/2/iv")    ### change dir for different movement
library(stringi)
library(stringr)

filename <- "op20n2-04_mel1_features.krn"                                                 ######CHANGE FILE!!
datafile<-lapply(filename,read.delim, header=F, stringsAsFactors=F)
dataframe <- datafile[[1]]

colnames(dataframe)=c("harm", "melody", "index", "beat_pos","duration","approaching")

### Delete null point rows and investigation
dataframe<-dataframe[!(dataframe$duration==0),] #remove dur = 0 data (originally '.' in pieces),they are generated as placeholders to '.'
#also including GRACE NOTE and rests
dataframe<-dataframe[!(dataframe$duration=='.'),] #rests and unknown notes with intervals are in [] form, either starting/end of piece
#so it does not affect intervals 
dataframe<-dataframe[!(dataframe$beat_pos=='.'),] #some rests

##if two or more note playing at the same time, process intervals to select interval the lower note (taking lower note as melody)
for (i in 1:nrow(dataframe)){
  int_set <- dataframe$approaching[i]
  dataframe$approaching[i]<-str_split(int_set, " ")[[1]][1]
}


#moving up approaching interval to get departure interval
#placeholder
app<-c(dataframe$approaching)
a<- app[-1]
a<-append(a,'[cc]')
dataframe$depart <- a

### "duration" as numeric
dataframe$duration <-as.numeric(dataframe$duration)

#####################Generate other factor cols##############################################################################
######linearize intervals#########
arrnumber <- c("ArrNum") 
dataframe[,arrnumber] <- NA 
depnumber <- c("DepNum") 
dataframe[,depnumber] <- NA 


intervals <- c('-A11'= -18,'-A12'=-20,'-A15'=-25,'-A18'=-30,'-A2'=-3,'-A4'=-6,'-A5'=-8,'-A8'=-13,'-A9'=-15,'-d10'=-14,'-d11'=-16,'-d12'=-18,'-d14'=-21,'-d15'=-23,'-d18'=-28,
               '-d3'=-2,'-d4'=-4,'-d5'=-6 ,'-d6'=-7,'-d7'=-9,'-d8'=-11,'-m10'=-15,'-M10'=-16,'-m13'=-20,'-M13'=-21,'-m14'=-22,'-M14'=-23,'-m16'=-25,'-M16'=-26,'-m17'=-27, 
               '-M17'=-28,'-m2'=-1,'-M2'=-2,'-M21'=-35,'-M24'=-40,'-m3'=-3,'-M3'=-4,'-m6'=-8,'-M6'=-9,'-m7'=-10,'-M7'=-11,'-m9'=-13,'-M9'=-14,'-P11'=-17,'-P12'=-19, 
               '-P15'=-24,'-P18'=-29,'-P19'=-31,'-P22'=-36,'-P4'=-5,'-P5'=-7,'-P8'=-12, 
               '+A11'= 18, '+A13'=22,'+A2'=3,'+A3'=5,'+A4'=6,'+A5'=8,'+A8'=13,'+A9'=15,'+d12'=18,'+d15'=23,'+d21'=33,'+d3'=2,'+d4'=4,'+d5'=6, 
               '+d7'=9,'+d8'=11,'+dd11'=15,'+m10'=15,'+M10'=16,'+m13'=20,'+M13'=21,'+m14'=22,'+M14'=23,'+m16'=25,'+M16'=26,'+m17'=27,'+M17'=28,'+m2'=1,'+M2'=2, 
               '+M20'=33,'+m21'=34,'+M21'=35,'+m23'=37,'+M28'=47,'+m3'=3,'+M3'=4,'+m6'=8,'+M6'=9,'+m7'=10,'+M7'=11,'+m9'=13,'+M9'=14,'+P11'=17, 
               '+P12'=19,'+P15'=24,'+P18'=29,'+P19'=31,'+P22'=36,'+P25'=41,'+P26'=43,'+P29'=48,'+P4'=5,'+P5'=7,'+P8'=12,'A1'=1,'AA1'=2,'d1'=-1,'P1'=0)


for (i in 1:nrow(dataframe)){
  dataframe$ArrNum[i]<-as.numeric(intervals[as.character(dataframe[i,]$approaching)])
  dataframe$DepNum[i]<-as.numeric(intervals[as.character(dataframe[i,]$depart)])
}


#fill missing interval data with 0 (first/last note in the piece)
dataframe$ArrNum[is.na(dataframe$ArrNum)] <- 0
dataframe$DepNum[is.na(dataframe$DepNum)] <- 0

data <- dataframe

####"onOffBeat" - beat_pos: onBeat/offBeat 
onOffBeat <- c("onOffBeat") #an empty list record result
dataframe[,onOffBeat] <- NA 
for (i in 1:nrow(data)){
  if (as.numeric(data$beat_pos[i]) %% 1 == 0){
    dataframe$onOffBeat[i] <- "onbeat"
  }else{
    dataframe$onOffBeat[i] <- "offbeat"
  }
}
dataframe$onOffBeat <-factor(dataframe$onOffBeat)

#### "ArrSkiSteLeap" & "DepSkiSteLeap" - Intervals: Skip, leap - ##########
arrSkiSteLeap<- c("ArrSkiSteLeap") #an empty list record result
dataframe[,arrSkiSteLeap] <- NA 
depSkiSteLeap<- c("DepSkiSteLeap") #an empty list record result
dataframe[,depSkiSteLeap] <- NA 
for (i in 1:nrow(data)){
  if (abs(as.numeric(data$ArrNum[i]))==2 | abs(as.numeric(data$ArrNum[i]))==1){
    dataframe$ArrSkiSteLeap[i] <- "step"
  } else if (abs(as.numeric(data$ArrNum[i]))==3 | abs(as.numeric(data$ArrNum[i]))==4){
    dataframe$ArrSkiSteLeap[i] <- "leap"
  } else if (abs(as.numeric(data$ArrNum[i]))==0){
    dataframe$ArrSkiSteLeap[i] <- "unison"
  } else {
    dataframe$ArrSkiSteLeap[i] <- "leap"
  }
  
  if (abs(as.numeric(data$DepNum[i]))==2 | abs(as.numeric(data$DepNum[i]))==1){
    dataframe$DepSkiSteLeap[i] <- "step"
  } else if (abs(as.numeric(data$DepNum[i]))==3 | abs(as.numeric(data$DepNum[i]))==4){
    dataframe$DepSkiSteLeap[i] <- "leap"
  } else if (abs(as.numeric(data$DepNum[i]))==0){
    dataframe$DepSkiSteLeap[i] <- "unison"
  } else {
    dataframe$DepSkiSteLeap[i] <- "leap"
  }
}


#### prepare for identification in the next program  ##########
prediction <- c("Prediction") #an empty list record result
dataframe[,prediction] <- NA #this line can be placed at the end 

response <- c("Response") # empty list record chord tone/non chord tone response
dataframe[,response] <- NA


###################MODEL PREDICTION##################################################################################

# make prediction on the melody data
dataframe$Prediction<-predict(model, newdata = dataframe, 'response')
# set threshold: >=0.5 as chord tones, <0.5 as non chord tones
for (i in 1:nrow(dataframe)){
  if (as.numeric(dataframe$Prediction[i]) >=0.5){ ###CHANGE THRESHOLD!
    dataframe$Response[i] <- 1
  }else{
    dataframe$Response[i] <- 0
  }
}
#write.csv(dataframe,'mel1.csv')                                                                     #####CHANGE NAME
########LOAD & MARK ORIGIN SCORE##############################################################################################################

raw <- "raw_score.krn"                                                                                #####CHANGE FILE
#count.fields(raw, sep = "\t")
rawfile<-read.delim2(raw, header=F, stringsAsFactors=F, sep = "\t", quote="\"")
colnames(rawfile)=c("harm","four", "three", "two","one", "index")

##search NCT and change to a REST in org score
for(i in 1:nrow(dataframe)){ #for loop counting dataframe
  if (dataframe$Response[i] == 0){
    num <- which(rawfile[,6] == dataframe$index[i])
    #find the row and change to "r"
    note<-rawfile$one[num]                                                                               ####CHANGE COLUMN!!
    tmp<-str_remove_all(note,"[^0-9.]")
    rawfile$one[num]<- paste(tmp,"r",sep='')                                                            ####CHANGE COLUMN!!
  }
}

#drop the index column
#final<-rawfile[ , !(names(rawfile) %in% 'index')]


write.table(rawfile, file = "score_reduced1.krn", sep = "\t",
            col.names=FALSE, row.names=FALSE, quote = FALSE)                                           ####CHANGE filename!!



