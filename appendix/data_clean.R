# Load necessary libraries
library(dplyr)
library(readxl)

# Load the data from the Excel file
survey_data <- read_excel("DATA2x02_Survey.xlsx")

# 1. Remove irrelevant columns (e.g., Timestamp if not needed)
survey_data_clean <- survey_data[,c(
  'What final grade are you aiming to achieve in DATA2x02?',
  'When it comes to assignments / due tasks do you:',
  'Would you prefer to have a trimester system (3 main teaching sessions per year) or stick with the existing semester system (2 main teaching sessions per year)?',
  'Do you tend to lean towards saying "yes" or towards saying "no" to things throughout life?',
  'Do you pay rent?',
  'What is the average amount of money you spend each week on food/beverages?',
  'What are your current living arrangements?',
  'How much alcohol do you consume each week?',
  'Do you believe in the existence of aliens (Extraterrestrial Life)?',
  'How often would you say you feel anxious on a daily basis?',
  'How many hours a week do you spend studying?',
  'Do you work?',
  'How consistent would you rate your sleep schedule?',
  'What is your diet style?',
  'Pick a number at random between 1 and 10 (inclusive)',
  'What is your favourite number (between 1 and 10 inclusive)',
  'Do you have a driver\'s license?',
  'Do you currently have a partner?',
  'What computer OS (operating system) are you currently using?',
  'How do you like your steak cooked?',
  'What is your dominant hand?',
  'Which unit are you enrolled in?',
  'Do you submit assignments on time?',
  'How many Weet-Bix would you typically eat in one sitting?',
  'Have you ever used R before starting DATA2x02?',
  'What kind of role (active or passive) do you think you are when working as part of a team?',
  'What is your WAM (weighted average mark)?',
  'On average, how many hours each week do you spend exercising?',
  'How many hours a week (on average) do you work in paid employment?', 
  'How much sleep do you get (on avg, per day)?')]

# 2. Convert text to numeric, and set invalid entries to NA, convert to time
sleep_data <- survey_data_clean$'How much sleep do you get (on avg, per day)?'

convert_time <- function (sleep_str) {
  
  clean_str <- gsub("[^0-9:.~\\-]", "", sleep_str)
  
  # If the value is a range ("6-8")
  if (grepl("-", clean_str)) {
    range_vals <- as.numeric(unlist(strsplit(clean_str, "-")))
    return(mean(range_vals))  # Return the midpoint
    
    # If the value is a range ("6~8")
  } else if (grepl("~", clean_str)) {
    range_vals <- as.numeric(unlist(strsplit(clean_str, "~")))
    return(mean(range_vals))  # Return the midpoint
    
    # If the value is a range ("8:30")
  } else if (grepl(":", clean_str)){
    time_parts <- as.numeric(unlist(strsplit(clean_str, ":")))
    hour_vals <- time_parts[1]
    minute_vals <- time_parts[2] / 60
    return(as.numeric(hour_vals + minute_vals))
    
    # If it's just a single numeric value in string form ("8")
  } else {
    return(as.numeric(clean_str))  # Convert single value to numeric
  }
}

# detecting the unit given and round all entries to 2 decimal places, ensure consistent format
convert_sleep <- function(sleep_str) {
  if (grepl("h|H", sleep_str) && grepl("m|M", sleep_str)){
    converted_str <- gsub("([0-9]+)\\s*[hH]* [^0-9]*([0-9]+)\\s*[mM]* [^0-9]*", "\\1:\\2", sleep_str)
    # s*: any space; [hH]*: starts with h or H; [mM]*: starts with m or M
    final_val <- round(convert_time(converted_str), 2)
    return(final_val)
    
  } else if (grepl("h", sleep_str) || grepl("H", sleep_str)) {
    final_val <- round(convert_time(sleep_str), 2)
    return(final_val)
    
  } else if (grepl("m|M", sleep_str) || grepl("mins|minutes", sleep_str)) {
    final_val <- round(convert_time(sleep_str) / 60, 2)
    return(final_val)
    
  } else {
    final_val <- round(convert_time(sleep_str), 2)
    return(final_val)
  }
}

# Apply the function to the sleep data
survey_data_clean$'How much sleep do you get (on avg, per day) in hours?' <- sapply(sleep_data, convert_sleep)
survey_data_clean <- survey_data_clean %>%
  select(-`How much sleep do you get (on avg, per day)?`)


# convert other columns to numeric
survey_data_clean <- survey_data_clean %>%
  mutate(`What is your WAM (weighted average mark)?` = as.numeric(`What is your WAM (weighted average mark)?`))  %>%
  mutate(`What is the average amount of money you spend each week on food/beverages?` = as.numeric(`What is the average amount of money you spend each week on food/beverages?`))  %>%
  mutate(`On average, how many hours each week do you spend exercising?` = as.numeric(`On average, how many hours each week do you spend exercising?`))  %>%
  mutate(`How many hours a week (on average) do you work in paid employment?` = as.numeric(`How many hours a week (on average) do you work in paid employment?`))  %>%
  mutate(`How many hours a week do you spend studying?` = as.numeric(`How many hours a week do you spend studying?`)) %>%
  mutate(`How many Weet-Bix would you typically eat in one sitting?` = as.numeric(`How many Weet-Bix would you typically eat in one sitting?`))

# convert the text responses for "other choices" into na due to limited response for this type
valid_choices <- list(
  "What is your diet style?" = c("Vegan", "Vegetarian", "Pescatarian", "Omnivorous"),
  "What computer OS (operating system) are you currently using?" = c("Windows", 
                                                                     "MacOS", 
                                                                     "Linux"),
  "Would you prefer to have a trimester system (3 main teaching sessions per year) or stick with the existing semester system (2 main teaching sessions per year)?" = c("Trimester", 
                                                                                     "Semester"),
  "Do you currently have a partner?" = c("Yes", "No"),
  "What are your current living arrangements?" = c("With parent(s) and/or sibling(s)",
                                                   "With partner", "College or student accomodation",
                                                   "Alone", "Share house"),
  "Do you pay rent?" = c("Yes", "No"),
  "Do you have a driver's license?" = c("Yes", "No"),
  "How do you like your steak cooked?" = c("Rare", "Medium-rare", "Medium", "Medium-well done",
                                           "Well done", "I don't eat beef")
)

convert_to_na <- function(column, valid_values) {
  column <- ifelse(is.na(column), NA, ifelse(column %in% valid_values, column, NA))
  return(column)
}


survey_data_clean <- survey_data_clean %>%
  mutate(across(names(valid_choices), 
                ~ convert_to_na(.x, valid_choices[[cur_column()]])))

# 3. Handle missing values
# Remove rows where more than 5 columns have missing values
survey_data_clean <- survey_data_clean %>%
  filter(rowSums(is.na(.)) < 5)

# 4. Handle outliers
survey_data_clean <- survey_data_clean %>%
  mutate(`What is your WAM (weighted average mark)?` = ifelse(`What is your WAM (weighted average mark)?` > 30, `What is your WAM (weighted average mark)?`, NA)) %>%
  mutate(`What is the average amount of money you spend each week on food/beverages?` = ifelse(`What is the average amount of money you spend each week on food/beverages?` > 600, NA, `What is the average amount of money you spend each week on food/beverages?`)) %>%
  mutate(`On average, how many hours each week do you spend exercising?` = ifelse(`On average, how many hours each week do you spend exercising?` > 50, NA, `On average, how many hours each week do you spend exercising?`)) %>%
  mutate(`How many hours a week (on average) do you work in paid employment?` = ifelse(`How many hours a week (on average) do you work in paid employment?` > 100, NA, `How many hours a week (on average) do you work in paid employment?`)) %>%
  mutate(`How much sleep do you get (on avg, per day) in hours?` = ifelse(`How much sleep do you get (on avg, per day) in hours?` < 4, NA, `How much sleep do you get (on avg, per day) in hours?`)) %>%
  mutate(`How many hours a week do you spend studying?` = ifelse(`How many hours a week do you spend studying?` > 100, NA, `How many hours a week do you spend studying?`)) %>%
  mutate(`How many Weet-Bix would you typically eat in one sitting?` = ifelse(`How many Weet-Bix would you typically eat in one sitting?` > 30, NA, `How many Weet-Bix would you typically eat in one sitting?`)) 

# Add a categorical variable of WAM grade band after cleaning outliers
survey_data_clean$'What is your WAM grade band?' <- cut(
  survey_data_clean$'What is your WAM (weighted average mark)?',
  breaks = c(50, 65, 75, 85, 100), # since from data description, the minimum value in WAM is 50.
  labels = c("PS", "CR", "DI", "HD"),
  right = FALSE
)

# 5. Convert categorical columns to factors
# Identify categorical columns (those with character or factors)
categorical_cols <- sapply(survey_data_clean, is.character)
survey_data_clean[categorical_cols] <- lapply(survey_data_clean[categorical_cols], as.factor)

# 6. View summary of cleaned data
summary(survey_data_clean)

# 7. Save the cleaned data to a new CSV file
write.csv(survey_data_clean, "cleaned_survey_data.csv", row.names = FALSE)

# Optional: Display the first few rows of the cleaned data
head(survey_data_clean)

