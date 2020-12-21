rm(list = ls())

setwd("/Users/tianxuehu/documents/ChordSymbolRec/datasets/ABC/quartets-v2-humdrum")
library(stringi)
library(stringr)

files=dir(pattern="*reduced4.krn$")

for (file in files){
  datafile<-lapply(file,read.delim, header=F, stringsAsFactors=F)
  dataframe <- datafile[[1]]
  cat('working on', file,'...\n')
  dataframe <- dataframe[1:(length(dataframe)-4)]
  
  cur_piece <- tools::file_path_sans_ext(file)
  save_name<- paste(cur_piece, '_tmp.krn',sep = '')
  write.table(dataframe, file = save_name, sep = "\t",
              col.names=FALSE, row.names=FALSE, quote = FALSE)  
}
