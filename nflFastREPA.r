# Load libraries
library(nflreadr)
library(dplyr)
library(tidyr)
library(writexl)

# Load datasets
pbp_2024 <- load_pbp(seasons = 2024)
schedules_2024 <- load_schedules(seasons = 2024)

# Clean PBP to valid plays only
pbp_clean <- pbp_2024 %>%
  filter(!is.na(play_type), !is.na(posteam), !is.na(defteam))

# ==== Calculate EPA Metrics Per Team Per Game ====
team_epa_stats <- pbp_clean %>%
  filter(!is.na(epa), !is.na(posteam)) %>%
  group_by(game_id, posteam) %>%
  summarise(
    EPA_per_play = mean(epa, na.rm = TRUE),
    PASS_EPA = mean(epa[play_type == "pass"], na.rm = TRUE),
    RUSH_EPA = mean(epa[play_type == "run"], na.rm = TRUE),
    SUCCESS_RATE = mean(success == 1, na.rm = TRUE),
    .groups = "drop"
  )

# ==== Core Team Game Stats ====
team_game_stats <- pbp_clean %>%
  group_by(game_id, posteam) %>%
  summarise(
    PASSINGYARDS = sum(passing_yards, na.rm = TRUE),
    RUSHINGYARDS = sum(rushing_yards, na.rm = TRUE),
    TURNOVERS = sum((interception == 1 | fumble_lost == 1), na.rm = TRUE),
    THIRDDOWN_ATT = sum(down == 3, na.rm = TRUE),
    THIRDDOWN_CONV = sum(down == 3 & first_down == 1, na.rm = TRUE),
    REDZONE_PLAYS = sum(yardline_100 <= 20, na.rm = TRUE),
    REDZONE_TD = sum(yardline_100 <= 20 & touchdown == 1, na.rm = TRUE),
    PENALTYYARDS = sum(penalty_yards, na.rm = TRUE),
    POINTSSCORED = max(posteam_score, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    `THIRDDOWN%` = ifelse(THIRDDOWN_ATT > 0, THIRDDOWN_CONV / THIRDDOWN_ATT, NA),
    `REDZONETD%` = ifelse(REDZONE_PLAYS > 0, REDZONE_TD / REDZONE_PLAYS, NA)
  ) %>%
  select(-THIRDDOWN_ATT, -THIRDDOWN_CONV, -REDZONE_PLAYS, -REDZONE_TD)

# ==== Merge EPA Stats ====
team_game_stats <- team_game_stats %>%
  left_join(team_epa_stats, by = c("game_id", "posteam"))

# ==== Add HOME Column ====
home_away_lookup <- schedules_2024 %>%
  select(game_id, home_team, away_team) %>%
  pivot_longer(cols = c(home_team, away_team),
               names_to = "location", values_to = "team") %>%
  mutate(HOME = ifelse(location == "home_team", 1, 0)) %>%
  select(-location)

team_game_stats <- team_game_stats %>%
  left_join(home_away_lookup, by = c("game_id" = "game_id", "posteam" = "team"))

# ==== Add OPPONENT Column and OPPONENTPAG ====
opponent_lookup <- home_away_lookup %>%
  rename(opponent = team)

team_game_stats <- team_game_stats %>%
  left_join(opponent_lookup, by = "game_id") %>%
  filter(opponent != posteam)

opponent_points <- team_game_stats %>%
  group_by(opponent) %>%
  summarise(OPPONENTPAG = mean(POINTSSCORED, na.rm = TRUE), .groups = "drop")

# ==== Final Merge and Clean ====
final_df <- team_game_stats %>%
  left_join(opponent_points, by = "opponent") %>%
  rename(TEAM = posteam)

final_df <- final_df %>%
  select(
    TEAM, game_id, HOME = HOME.x, PASSINGYARDS, RUSHINGYARDS, TURNOVERS,
    `THIRDDOWN%`, `REDZONETD%`, PENALTYYARDS, OPPONENTPAG,
    EPA_per_play, PASS_EPA, RUSH_EPA, SUCCESS_RATE,
    POINTSSCORED
  )

# ==== Save to Excel ====
write_xlsx(final_df, path = "nfl_2024_team_game_stats_EPA.xlsx")

# ==== Preview ====
print(head(final_df))
