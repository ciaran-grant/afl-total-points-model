# Load Library
library(AFL)
library(tidyverse)
library(plyr)
library(data.table)
# Load in Stats object
Stats = load_stats()
# Specify years
seasons <- c("2009","2010","2011","2012","2013",
             "2014","2015","2016","2017","2018",
             "2019","2020","2021", "2022")

round_metadata_list = c(".consolidated_metadata",".__enclos_env__","Metadata","clone","initialize")

aus_sports_betting_odds <- list()

for (season in seasons){
  print(season)
  round_list = sort(names(Stats)[grepl(paste0("^",season), names(Stats))])
  for (round in round_list){
    print(round)
    Stats_round = Stats[[round]]
    match_list= sort(setdiff(names(Stats_round), round_metadata_list))
    for (match in match_list){
      Stats_match = Stats_round[[match]]
      print(match)
      aus_sports_betting_odds = rbind.fill(aus_sports_betting_odds, Stats_match[['AusSportsBetting_Histoical_Odds']])
      }
  }
}

aus_sports_betting_odds <- data.table(aus_sports_betting_odds)


# Save data
saveRDS(aus_sports_betting_odds, "/total-points-score-model/data/aus_sports_betting_odds")
write.csv(aus_sports_betting_odds, "/total-points-score-model/data/aus_sports_betting_odds.csv")
