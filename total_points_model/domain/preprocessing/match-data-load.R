# Load Library
library(AFL)
library(tidyverse)
library(plyr)
library(data.table)
# Load in Stats object
Stats = load_stats()
# Specify years
seasons <- c("2005","2006","2007","2008","2009","2010","2011","2012","2013",
             "2014","2015","2016","2017","2018","2019","2020","2021", "2022")

round_metadata_list = c(".consolidated_metadata",".__enclos_env__","Metadata","clone","initialize")

afl_api_match_stats <- list()
afl_tables_match_stats <- list()
fryzigg_match_stats <- list()

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
      afl_api_match_stats = rbind.fill(afl_api_match_stats, Stats_match[['AFL_API_Fixture_And_Results']])
      afl_tables_match_stats = rbind.fill(afl_tables_match_stats, Stats_match[['AFLTables_Match_Summary']])
      fryzigg_match_stats = rbind.fill(fryzigg_match_stats, Stats_match[['Fryzigg_Match_Summary']])
      }
    
  }
}

afl_api_match_stats <- data.table(afl_api_match_stats)
afl_tables_match_stats <- data.table(afl_tables_match_stats)
fryzigg_match_stats <- data.table(fryzigg_match_stats)

# Merge into single dataset
afl_tables_cols <- c("Match_ID", setdiff(names(afl_tables_match_stats), names(afl_api_match_stats)))
match_data <- merge(afl_api_match_stats, afl_tables_match_stats[, ..afl_tables_cols], by = "Match_ID")
fryzigg_cols <- c("Match_ID", setdiff(names(fryzigg_match_stats), names(match_data)))
match_data <- merge(match_data, fryzigg_match_stats[, ..fryzigg_cols], by = "Match_ID")

# Save data
write.csv(match_data, "/data/merged-data/afl_match_data.csv", row.names = FALSE)
