# Load Library
library(AFL)
library(tidyverse)
library(plyr)
library(data.table)
# Load in Stats object
Stats = load_stats()

# Get match_stats
match_stats = Stats$Match_Stats()

# Rename
setnames(match_stats, old = c('Total Game Score', "Home Win"), new = c('Total_Game_Score', "Home_Win"))

# Merge on Home Ground info - is Home Team actually playing at Home?
venue_info = Stats$Venues
venue_info = venue_info[, .(Venue, Ground_Width, Ground_Length)]
match_stats <- merge(match_stats, venue_info, by = "Venue")

# Merge on Venue information - length and width
team_info = Stats$Team_Info
team_info = team_info[, .(Team, Home_Ground_1, Home_Ground_2, Home_Ground_3)]
match_stats <- merge(match_stats, team_info, by.x = "Home_Team", by.y = "Team")

match_stats[is.na(Home_Ground_2), Home_Ground_2 := ""]
match_stats[is.na(Home_Ground_3), Home_Ground_3 := ""]
match_stats[, Home_Ground := ifelse(Venue == Home_Ground_1, "Primary Home",
                                    ifelse(Venue == Home_Ground_2, "Secondary Home",
                                           ifelse(Venue == Home_Ground_3, "Tertiary Home",
                                                 "Neutral")))]
# Remove unnecessary columns
match_stats[, c("Home_Ground_1", "Home_Ground_2", "Home_Ground_3") := NULL]
match_stats[, c("Home_Odds_Close", "Away_Odds_Close",        
                "Home_Line_Close", "Away_Line_Close",       
                "Home_Line_Odds_Close", "Away_Line_Odds_Close",   
                "Total_Score_Close", "Total_Score_Over_Close", 
                "Total_Score_Under_Close", "Umpires") := NULL]

# Create ModellingFilter
match_stats[, DateTime := as_datetime(Date)]
match_stats[, Date := as_date(Date)]

match_stats[, ModellingFilter := ifelse(Date < '2019-01-01', TRUE, FALSE)]

# Sort by Date
setorder(match_stats, cols = "DateTime")

# Save data
write.csv(match_stats, "../afl_match_stats.csv", row.names = FALSE)
