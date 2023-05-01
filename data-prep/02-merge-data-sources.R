# Load Library
library(AFL)
library(tidyverse)
library(plyr)
library(data.table)

# Load AFL Data
afl_api_match_data <- readRDS("/total-points-score-model/data/raw-data/afl_api_match_stats.RDS")
afl_tables_match_data <- readRDS("/total-points-score-model/data/raw-data/afl_tables_match_stats.RDS")
footywire_match_data<- readRDS("/total-points-score-model/data/raw-data/footywire_match_stats.RDS")
fryzigg_match_data <- readRDS("/total-points-score-model/data/raw-data/fryzigg_match_stats.RDS")

# Seasons - Footywire starts from 2007
footywire_match_data[, .N, Year]

afl_api_match_data <- afl_api_match_data[Year >= 2007]
afl_tables_match_data <- afl_tables_match_data[Year >= 2007]
fryzigg_match_data <- fryzigg_match_data[Year >= 2007]

# Rounds
afl_api_rounds <- afl_api_match_data[, .N, c("Year", "Round_ID")]
afl_tables_rounds <- afl_tables_match_data[, .N, c("Year", "Round_ID")]
footywire_rounds <- footywire_match_data[, .N, c("Year", "Round_ID")]
fryzigg_rounds <- fryzigg_match_data[, .N, c("Year", "Round_ID")]

# Match_ID
length(afl_api_match_data[, unique(Match_ID)])
length(afl_tables_match_data[, unique(Match_ID)])
length(footywire_match_data[, unique(Match_ID)])
length(fryzigg_match_data[, unique(Match_ID)])

# Remove duplicates
afl_api_match_data <- unique(afl_api_match_data, by = "Match_ID")
afl_tables_match_data <- unique(afl_tables_match_data, by = "Match_ID")
footywire_match_data <- unique(footywire_match_data, by = "Match_ID")
fryzigg_match_data <- unique(fryzigg_match_data, by = "Match_ID")

sum(afl_api_match_data[, unique(Match_ID)] == afl_tables_match_data[, unique(Match_ID)]) == nrow(afl_api_match_data)
sum(afl_api_match_data[, unique(Match_ID)] == footywire_match_data[, unique(Match_ID)]) == nrow(afl_api_match_data)
sum(afl_api_match_data[, unique(Match_ID)] == fryzigg_match_data[, unique(Match_ID)]) == nrow(afl_api_match_data)

# Merge into single dataset
afl_tables_cols <- c("Match_ID", setdiff(names(afl_tables_match_data), names(afl_api_match_data)))
match_data <- merge(afl_api_match_data, afl_tables_match_data[, ..afl_tables_cols], by = "Match_ID")
footywire_cols <- c("Match_ID", setdiff(names(footywire_match_data), names(match_data)))
match_data <- merge(match_data, footywire_match_data[, ..footywire_cols], by = "Match_ID")
fryzigg_cols <- c("Match_ID", setdiff(names(fryzigg_match_data), names(match_data)))
match_data <- merge(match_data, fryzigg_match_data[, ..fryzigg_cols], by = "Match_ID")


## Create Home and Away scores
match_data[, c("Home_Score", "Away_Score") := tstrsplit(Q4_Score, " - ", fixed=TRUE)]
match_data[, c("Home_Goals", "Home_Behinds", "Home_Total_Score") := tstrsplit(Home_Score, ".", fixed = TRUE)]
match_data[, c("Away_Goals", "Away_Behinds", "Away_Total_Score") := tstrsplit(Away_Score, ".", fixed = TRUE)]

convert_to_integer = c("Home_Goals", "Home_Behinds", "Home_Total_Score", "Away_Goals", "Away_Behinds", "Away_Total_Score")
match_data[, c(convert_to_integer) := lapply(.SD, as.integer), .SDcols=convert_to_integer]

# Rename
setnames(match_data, old = c('Total Game Score', "Home Win"), new = c('Total_Game_Score', "Home_Win"))

### Modelling Data V1.0
saveRDS(match_data, "/total-points-score-model/data/modelling-data/total_score_match_data_v01.RDS")
