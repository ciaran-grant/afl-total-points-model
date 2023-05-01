###### Data Preparation - All Sources ######

# Load Library
library(AFL)
library(tidyverse)
library(plyr)
library(data.table)

##### Match Level Data ---------------------

# Load match data
match_data <- setDT(readRDS("/total-points-score-model/data/modelling-data/total_score_match_data_v01.RDS"))

match_data[, random5 := sample(5, size = nrow(match_data), replace = TRUE)]
match_data[, training_set := ifelse(random5 == 5, FALSE, TRUE)]

# What does the response look like? Distribution, min, max.
#  How many total points in a game?
match_data[, summary(Total_Game_Score)]

## Create Team Level Goals, Behinds and Total Scores
home_teams = match_data[, unique(Home_Team)]
away_teams = match_data[, unique(Away_Team)]
teams = unique(c(home_teams, away_teams))

convert_to_team_level_stats <- function(data, team){
  
  team_data = data[(Home_Team == team) | (Away_Team == team)]
  team_data[, Team := (ifelse(Home_Team == team, Home_Team, Away_Team))]
  team_data[, Home_Away := (ifelse(Home_Team == team, "Home", "Away"))]
  team_data[, For_Goals := (ifelse(Home_Team == team, Home_Goals, Away_Goals))]
  team_data[, For_Behinds := (ifelse(Home_Team == team, Home_Behinds, Away_Behinds))]
  team_data[, For_Total_Score := (ifelse(Home_Team == team, Home_Total_Score, Away_Total_Score))]
  team_data[, Opp := (ifelse(Home_Team == team, Away_Team, Home_Team))]
  team_data[, Against_Goals := (ifelse(Home_Team == team, Away_Goals, Home_Goals))]
  team_data[, Against_Behinds := (ifelse(Home_Team == team, Away_Behinds, Home_Behinds))]
  team_data[, Against_Total_Score := (ifelse(Home_Team == team, Away_Total_Score, Home_Total_Score))]
  
}

team_match_data <- data.table()
for (team in teams){
  print(team)
  team_data <- convert_to_team_level_stats(match_data, team)
  team_match_data <- rbind(team_match_data, team_data)
}
team_match_data <- data.table(team_match_data)[order(Date)]
team_match_data <- unique(team_match_data, by = c("Match_ID", "Team"))

team_match_data[, For_Scores := For_Goals + For_Behinds]
team_match_data[, Against_Scores := Against_Goals + Against_Behinds]

numeric_team_stats <- c(
  "Total_Game_Score",
  "For_Goals", 
  "For_Behinds",
  "For_Scores",
  "For_Total_Score",     
  "Against_Goals",
  "Against_Behinds",
  "Against_Scores",
  "Against_Total_Score" 
)
team_match_data[, (paste0(numeric_team_stats, "_avg2")) :=lapply(.SD, function(x) frollmean(x, 2, fill=NA, na.rm = T)), Team, .SDcols=numeric_team_stats]
# team_match_data[, (paste0(numeric_team_stats, "_avg3")) :=lapply(.SD, function(x) frollmean(x, 3, fill=NA, na.rm = T)), Team, .SDcols=numeric_team_stats]
# team_match_data[, (paste0(numeric_team_stats, "_avg5")) :=lapply(.SD, function(x) frollmean(x, 5, fill=NA, na.rm = T)), Team, .SDcols=numeric_team_stats]
team_match_data[, (paste0(numeric_team_stats, "_avg10")) :=lapply(.SD, function(x) frollmean(x, 10, fill=NA, na.rm = T)), Team, .SDcols=numeric_team_stats]


# ### some missings still
# team_match_data_non_rolling = team_match_data[145:nrow(team_match_data)]
# colSums(is.na(team_match_data_non_rolling))
# team_match_data_non_rolling[is.na(Total_Game_Score_avg2)]
# ### Gold Coast joined in 2011
# team_match_data[Team == "Gold Coast", mean(Total_Game_Score), c(Year, Team)]
# ### Western Sydney joined in 2012
# team_match_data[Team == "Western Sydney", mean(Total_Game_Score), c(Year, Team)]
# team_by_year <- team_match_data[, mean(Total_Game_Score), c('Year', 'Team')]
# 
# team_match_data_post2013 <- team_match_data[Year>2012]
# colSums(is.na(team_match_data_post2013)) # no missings anymore

## Drop features that aren't known at modelling time
names(team_match_data)

cols_to_drop <- c(
   "Q1_Score"                              
  ,"Q2_Score"                               
  ,"Q3_Score" 
  ,"Q4_Score"
  ,"Q5_Score"
  ,"Margin"                                
  ,"Home_Win"
  ,"Home_Score"
  ,"Away_Score"
  ,"Home_Goals"                           
  ,"Home_Behinds"                           
  ,"Home_Total_Score"                       
  ,"Away_Goals"                
  ,"Away_Behinds"                           
  ,"Away_Total_Score"                       
  ,"For_Goals"                           
  ,"For_Behinds"                           
  ,"For_Total_Score"
  ,"Against_Goals"                
  ,"Against_Behinds"                            
  ,"Against_Total_Score"                        
  ,"For_Scores"                    
  ,"Against_Scores"  
)

team_match_data <- team_match_data[, c(cols_to_drop) := NULL]

# Impute NAs with mean of column from training set.
numeric_training_data <- team_match_data[training_set == TRUE, .SD, .SDcols = is.numeric]
numeric_data <- team_match_data[, .SD, .SDcols = is.numeric]

for (i in seq_along(numeric_data)) set(numeric_data, i=which(is.na(numeric_data[[i]])), j=i, value=mean(numeric_training_data[[i]], na.rm=TRUE))

categorical_data <- team_match_data[ , .SD, .SDcols = is.character]
categorical_data[is.na(categorical_data), ] <- "Unknown" 

team_match_data <- cbind(categorical_data, numeric_data)

# Remove Covid 2020 Year
team_match_data = team_match_data[Year != 2020]
match_data = match_data[Year != 2020]

# Reduce back to single row per match

numeric_team_stats_avg = c(paste0(numeric_team_stats,"_avg2"),
                           paste0(numeric_team_stats,"_avg10"))

home_data = team_match_data[Home_Away == "Home"]
home_numeric_team_stats = paste0("Home_", numeric_team_stats_avg)
setnames(home_data, old = c(numeric_team_stats_avg), 
                    new = c(home_numeric_team_stats))

home_merge_cols = c("Match_ID", home_numeric_team_stats)
home_stats = home_data[, ..home_merge_cols]

match_data <- match_data %>% left_join( home_stats, 
                      by=c('Match_ID'))

away_data = team_match_data[Home_Away == "Away"]
away_numeric_team_stats = paste0("Away_", numeric_team_stats_avg)
setnames(away_data, old = c(numeric_team_stats_avg), 
         new = c(away_numeric_team_stats))

away_merge_cols = c("Match_ID", away_numeric_team_stats)
away_stats = away_data[, ..away_merge_cols]

match_data <- match_data %>% left_join( away_stats, 
                                        by=c('Match_ID'))

modelling_data_total_team_score <- match_data

# Save data
write.csv(modelling_data_total_team_score, "/total-points-score-model/data/modelling-data/modelling_data_total_team_score.csv")
saveRDS(modelling_data_total_team_score, "/total-points-score-model/data/modelling-data/modelling_data_total_team_score.RDS")

