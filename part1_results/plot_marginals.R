library(tidyverse)

theme_set(theme_bw() + theme(plot.title = element_text(face = 'bold', size = 18, hjust = 0.5),
                             panel.grid.major = element_line(color = 'grey70', size = 0.2),
                             panel.grid.minor = element_blank(),
                             axis.text.x = element_text(size=12,angle=0,hjust=.5,vjust=.5,face="plain"),
                             axis.text.y = element_text(size=12,hjust=.5,vjust=.5,face="plain"),  
                             axis.title.x = element_text(size=14,angle=0,hjust=.5,vjust=0,face="plain"),
                             axis.title.y = element_text(size=14,angle=90,hjust=.5,vjust=.5,face="plain"),
                             legend.title = element_text(size=14,face="plain"),
                             legend.text = element_text(size=14,face="plain")
) )

CatCrimes <- function(x)
{
  if(x == 'Prior Crimes = 0')
    return('0')
  else if(x == '1 <= Prior Crimes <= 3')
    return('{1,2,3}')
  else if(x == 'Prior Crimes > 3')
    return('> 3')
}

CatAge <- function(x)
{
  if(x == 'Age < 25')
    return('< 25')
  else if(x == '25 <= Age <= 45')
    return('[25,45]')
  else if(x == 'Age > 45')
    return('> 45')
}


df1 <- read_csv('preprocessing_compas_DP_unweighted_marginals.csv')
df2 <- read_csv('preprocessing_compas_DP_reweighted_marginals.csv')
df <- rbind(df1,df2)
df$X1 <- NULL
df[1,]$Original <- 'Unweighted'
df[2,]$Original <- 'Reweighted'
df$Original <- as.character(df$Original)
pd <- position_dodge(width = 0.2)

dfSex <- df %>% pivot_longer(c(`Sex = M`), names_to = "Buckets", values_to = "Freq") %>% select(c('Buckets','Freq','Original'))
dfSex$Feature <- rep('Sex',2)
dfSex$Buckets <- 'Male'

dfAge <- df %>% pivot_longer(c(`Age < 25`,`25 <= Age <= 45`,`Age > 45`), names_to = "Buckets", values_to = "Freq") %>% select(c('Buckets','Freq','Original'))
dfAge$Feature <- rep('Age',6)
dfAge$Buckets <- apply(dfAge[,c('Buckets')],1, CatAge)

dfCrimes <- df %>% pivot_longer(c(`Prior Crimes = 0`,`1 <= Prior Crimes <= 3`,`Prior Crimes > 3`), names_to = "Buckets", values_to = "Freq") %>% select(c('Buckets','Freq','Original'))
dfCrimes$Feature <- rep('Prior Crimes',6)
dfCrimes$Buckets <- apply(dfCrimes[,c('Buckets')],1, CatCrimes)

dfDegree <- df %>% pivot_longer(c(`Charge Degree = Felony`), names_to = "Buckets", values_to = "Freq") %>% select(c('Buckets','Freq','Original'))
dfDegree$Feature <- rep('Charge Degree',2)
dfDegree$Buckets <- 'Felony'

dfFeatures <- rbind(dfSex, dfAge, dfCrimes, dfDegree)

dfFeatures %>% ggplot() +
  geom_bar(aes(x=Buckets,y=Freq,fill=Original),stat='identity',position = pd) + facet_grid(.~Feature,scales = 'free_x', space = 'free_x') +
  scale_fill_brewer(palette = 'Set1') + ylab('Frequency in Data (%)') + theme(axis.title.x = element_blank(), legend.title = element_blank(), legend.position = 'top') +
  theme(strip.background = element_rect(colour = "black", fill = "white"))
  
  
  
  
  