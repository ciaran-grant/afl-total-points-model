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
afl_api_player_stats <- list()

afl_tables_match_stats <- list()
afl_tables_player_stats <- list()

footywire_match_stats <- list()
footywire_player_stats <- list()

fryzigg_match_stats <- list()
fryzigg_player_stats <- list()

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
      # print("AFL API Match Stats")
      afl_api_match_stats = rbind.fill(afl_api_match_stats, Stats_match[['AFL_API_Fixture_And_Results']])
      # print("AFL API Player Stats")
      afl_api_player_stats = rbind.fill(afl_api_player_stats, Stats_match[['AFL_API_Player_Stats']])
      
      # print("AFLTables Match Stats")
      afl_tables_match_stats = rbind.fill(afl_tables_match_stats, Stats_match[['AFLTables_Match_Summary']])
      # print("AFLTablesPlayer Stats")
      afl_tables_player_stats = rbind.fill(afl_tables_player_stats, Stats_match[['AFLTables_Player_Stats']])
      
      # print("Footywire Match Stats")
      footywire_match_stats = rbind.fill(footywire_match_stats, Stats_match[['Footywire_Match_Summary']])
      # print("Footywire Player Stats")
      footywire_player_stats = rbind.fill(footywire_player_stats, Stats_match[['Footywire_Player_Stats']])

      # print("Fryzigg Match Stats")
      fryzigg_match_stats = rbind.fill(fryzigg_match_stats, Stats_match[['Fryzigg_Match_Summary']])
      # print("Fryzigg Player Stats")
      fryzigg_player_stats = rbind.fill(fryzigg_player_stats, Stats_match[['Fryzigg_Player_Stats']])
    }
    
  }
}

afl_api_match_stats <- data.table(afl_api_match_stats)
afl_api_player_stats <- data.table(afl_api_player_stats)
afl_tables_match_stats <- data.table(afl_tables_match_stats)
afl_tables_player_stats <- data.table(afl_tables_player_stats)
footywire_match_stats <- data.table(footywire_match_stats)
footywire_player_stats <- data.table(footywire_player_stats)
fryzigg_match_stats <- data.table(fryzigg_match_stats)
fryzigg_player_stats <- data.table(fryzigg_player_stats)

# Save data
# saveRDS(afl_api_match_stats, "/total-points-score-model/data/afl_api_match_stats.RDS")
# saveRDS(afl_api_player_stats, "/total-points-score-model/data/afl_api_player_stats.RDS")
# saveRDS(afl_tables_match_stats, "/total-points-score-model/data/afl_tables_match_stats.RDS")
# saveRDS(afl_tables_player_stats, "/total-points-score-model/data/afl_tables_player_stats.RDS")
# saveRDS(footywire_match_stats, "/total-points-score-model/data/footywire_match_stats.RDS")
# saveRDS(footywire_player_stats, "/total-points-score-model/data/footywire_player_stats.RDS")
# saveRDS(fryzigg_match_stats, "/total-points-score-model/data/fryzigg_match_stats.RDS")
# saveRDS(fryzigg_player_stats, "/total-points-score-model/data/fryzigg_player_stats.RDS")