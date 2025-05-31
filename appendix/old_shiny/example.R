#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    https://shiny.posit.co/
#

library(shiny)
library(ggplot2) # to draw the plots for visualization 
library(dplyr) # for ltering, summarizing, and transforming data.
library(readr) # to read the csv file
library(broom) # convert statistical analysis objects (such as models or tests) into tidy data frames to be used in tidyverse
library(plotly) # to display interactive plots
library(DT) # to display tables
library(shinyjs) # to limit the number of selection choices

data <- read_csv("cleaned_survey_data.csv")
categorical_cols <- sapply(data, is.character)
data[categorical_cols] <- lapply(data[categorical_cols], as.factor)


# Define UI for application that draws a histogram
ui <- fluidPage(
  navbarPage(
    title = "DATA2x02 Survey Data Explore",
    tabPanel("About", 
             h3("About this App"),
             p("This Shiny app allows users to analyze the DATA2x02 survey data and perform hypothesis testing."),
             p("The survey data used in this app is the survey data responded by the DATA2x02 students."),
             p("The survey is not a random sample of DATA2X02. However, due to the purpose of hypothesis testing, it is assumed that the data is randomly sampled from the students, and the data are iid to each other."),
             p("Thus, this enables us to check the assumptions for different hypothesis tests."),
             p("It is not a random sample since it is a survey answered by volunteering students. Thus, the decision made by the students whether they want to participate removes randomness in the samping process."),
             HTML("<br><br>"),
             p("Users can select their interested variables for analysis. There are total 31 variables including 19 categorical variables and 12 numerical.",
             p("The chi-sqaure test is performed for two selected categorical variables. All the 19 categorical variables are in the drop down list for the selection."),
             p("This test has two types which are independent test and homogeneity test. "),
             p("The independent test is to determine whether the two variables are independent of each other or not."),
             p("The homogeneity test is to determine whether the distribution for a categorical variable (variable 1) is homogeneous for all the subgroups defined by another categorical variable (variable 2)."),
             HTML("<br><br>"),
             p("The two sample t-test is performed for one selected categorical variables to decide the two samples from the 19 categorical varibales, and the numerical variable is chosen from the 12 numerical variables in the drop down list."),
             p("This test is to determine the unknown population mean for the two samples (two groups) is equal or not by comparing the sample mean of the numerical variable for the two groups."),
             p("The result for this test would suggest whether one of the group would be superior than the other one in the field defined by the numerical variable."), 
             HTML("<br><br>"),
             p("I acknowledge the use of ChatGPT [https://chat.openai.com/] in fixing and debugging the code for the app. "),
             HTML("<br><br>"))
    ),
      
    # Hypothesis Testing Tab
    tabPanel("Hypothesis Testing (Chi-Square Test)",
             sidebarLayout(
               sidebarPanel(
                 selectInput("var1", "Select the first variable:", choices = names(data[categorical_cols])),
                 selectInput("var2", "Select the second variable:", choices = names(data[categorical_cols])),
                 selectInput("chi_type", "Select the test type:", choices = c("Independence Test", "Homogeneity Test")),
                 uiOutput("var1_levels")
               ),
               mainPanel(
                 tabsetPanel(
                   tabPanel(
                     "Variable Distribution plot", 
                     HTML("<br><br>"), # Adds two empty lines
                     plotly::plotlyOutput('dist_plot1'),
                     HTML("<br><br>"), # Adds two empty lines
                     plotly::plotlyOutput('dist_plot2'),
                     HTML("<br><br>"),
                     uiOutput("des_p3"),
                     plotly::plotlyOutput('dist_plot3'),
                     HTML("<br><br>")
                   ),
                   tabPanel(
                     "Table and Assumptions",
                     HTML("<br><br>"),
                     HTML("<strong>Contingency table with Categories from Variable 1 based on selected category of interest: </strong>"),
                     HTML("<br><br>"),
                     DT::DTOutput('table_chi'),
                     HTML("<br><br><br>"),
                     HTML("<strong>Frequency table based on the contingency table to show the expected values of Variable 2 values for the selected category of interest of Variable 1: </strong>"),
                     HTML("<br><br>"),
                     dataTableOutput('table_freq'),
                     HTML("<br><br>"),
                     uiOutput("assumption1")
                   ),
                   tabPanel("Testing Results", 
                            HTML("<br><br>"),
                            uiOutput("chi_interpretation"),
                            uiOutput("chiq_test"),
                            uiOutput("chi_conclusion"),
                            HTML("<br><br>")),
                   tabPanel("Graph", HTML("<br><br>"), uiOutput("graph_assum"),
                            HTML("<br><br>"),
                            plotly::plotlyOutput('graph_chi'), HTML("<br><br><br>"))
                 ))
               )
      ),
    
    tabPanel("Hypothesis test (Two Sample t-test, Welch Two Sample t-test if significant difference in sample variance)",
             sidebarLayout(
               sidebarPanel(
                 selectInput("cat1", "Select a categorical variable (variable 1):", choices = names(data[categorical_cols])),
                 selectInput("num2", "Select a numeric variable (variable 2):", choices = names(data)[sapply(data, is.numeric)]),
                 
                 selectInput("t_type", "Select alternative hypothese condition: ", choices = c("Greater", "Less", "Two Sided")),

                 uiOutput("cat1_levels")
               ),
               mainPanel(
                 tabsetPanel(
                   tabPanel(
                     "Variable Distribution plot", 
                     HTML("<br><br>"),
                     plotly::plotlyOutput('boxplot_output'),
                     HTML("<br><br>"),
                     plotly::plotlyOutput('hist_output'),
                     HTML("<br><br>"),
                     uiOutput('warn_text'),
                     HTML("<br><br>"),
                     plotly::plotlyOutput('side_by_side'),
                     HTML("<br><br>")
                   ),
                   tabPanel(
                     "Assumptions", HTML("<br><br>"),
                     uiOutput('t_assum'),
                     HTML("<br><br>"),
                     uiOutput("var_output"),
                     plotly::plotlyOutput('norm_qq'),
                     HTML("<br><br>"),
                     dataTableOutput("stats_table"),
                     HTML("<br><br>"),
                     uiOutput('desc_qq'),
                     HTML("<br><br>")
                   ),
                   tabPanel("Testing Results", 
                            HTML("<br><br>"),
                            uiOutput("var_output"),
                            verbatimTextOutput('t_test_result'),
                            HTML("<br><br>"),
                            uiOutput("t_test"),
                            HTML("<br><br>"),
                            uiOutput("t_conc"),
                            HTML("<br><br>")
                            ),
                   tabPanel("Graph", HTML("<br><br>"),
                            plotOutput('graph_t'), HTML("<br><br><br>"))
                 ))
             )
        )
    )
)

# if more than two categories in the variable
get_result <- function(data, A, B, C){
  cleaned_data <- data[!is.na(data[[A]]) & !is.na(data[[B]]), ]
  # Create a modified version of the dataset where non-selected categories are labeled as "Other"

  modified_data <- cleaned_data %>%
    mutate(new_category = ifelse(cleaned_data[[A]] %in% C, C, "Sum of Other Categories")) %>%
    droplevels()  # Drop unused levels to clean up the factor levels
  
  contingency_table <- table(modified_data$new_category, modified_data[[B]])
  #contingency_table <- table(cleaned_data[[A]], cleaned_data[[B]])
  
  return(contingency_table) 
}

# if only two categories in the variable
get_table <- function(data, A, B, C){
  cleaned_data <- data[!is.na(data[[A]]) & !is.na(data[[B]]), ]
  contingency_table <- table(cleaned_data[[A]], cleaned_data[[B]])
  
  return(contingency_table) 
}

above_5_chi <- function(data, A, B, C){
  var1_data <- na.omit(data[[A]])
  var1_levels <- unique(var1_data)
  
  if (length(var1_levels) == 2) {
    contingency_table_result <- get_table(data, A, B, C)
  } else {
    contingency_table_result <- get_result(data, A, B, C)
  }
  
  test_result <- chisq.test(contingency_table_result)
  expected_val <- round(test_result$expected, 2)
  
  # Check if all expected values are above 5
  all_above_5 <- all(expected_val > 5)
  
  return(all_above_5)

}

get_t_result <- function(data, A, B, C, type) {
  # Ensure exactly two categories are selected
  if (length(C) != 2) {
    return("Please select exactly two categories to perform the t-test.")
  }
  
  selected_data <- data[data[[A]] %in% C, ]
  selected_data <- na.omit(selected_data[, c(A, B)])
  
  categorical_var <- selected_data[[A]]
  numerical_var <- selected_data[[B]]
  
  # Perform the two-sample t-test
  if (type == "Two Sided"){
    t_test_result <- t.test(
      numerical_var ~ categorical_var,
      alternative = "two.sided", var.equal = TRUE
    )
  } else if (type == "Greater"){
    t_test_result <- t.test(
      numerical_var ~ categorical_var,
      alternative = "greater", var.equal = TRUE
    )
  } else if (type == "Less"){
    t_test_result <- t.test(
      numerical_var ~ categorical_var,
      alternative = "less", var.equal = TRUE
    )
  }
  
  return(t_test_result)
}
get_t_result <- function(data, A, B, C, type) {
  # Ensure exactly two categories are selected
  if (length(C) != 2) {
    return("Please select exactly two categories to perform the t-test.")
  }
  
  selected_data <- data[data[[A]] %in% C, ]
  selected_data <- na.omit(selected_data[, c(A, B)])
  
  categorical_var <- selected_data[[A]]
  numerical_var <- selected_data[[B]]
  
  # Perform the two-sample t-test
  if (type == "Two Sided"){
    t_test_result <- t.test(
      numerical_var ~ categorical_var,
      alternative = "two.sided", var.equal = TRUE
    )
  } else if (type == "Greater"){
    t_test_result <- t.test(
      numerical_var ~ categorical_var,
      alternative = "greater", var.equal = TRUE
    )
  } else if (type == "Less"){
    t_test_result <- t.test(
      numerical_var ~ categorical_var,
      alternative = "less", var.equal = TRUE
    )
  }
  
  return(t_test_result)
}

get_welch_result <- function(data, A, B, C, type) {
  # Ensure exactly two categories are selected
  if (length(C) != 2) {
    return("Please select exactly two categories to perform the t-test.")
  }
  
  selected_data <- data[data[[A]] %in% C, ]
  selected_data <- na.omit(selected_data[, c(A, B)])
  
  categorical_var <- selected_data[[A]]
  numerical_var <- selected_data[[B]]
  
  # Perform the two-sample t-test
  if (type == "Two Sided"){
    t_test_result <- t.test(
      numerical_var ~ categorical_var,
      alternative = "two.sided"
    )
  } else if (type == "Greater"){
    t_test_result <- t.test(
      numerical_var ~ categorical_var,
      alternative = "greater"
    )
  } else if (type == "Less"){
    t_test_result <- t.test(
      numerical_var ~ categorical_var,
      alternative = "less"
    )
  }
  
  return(t_test_result)
}

check_variance <- function(data, A, B, C){
  # Ensure exactly two categories are selected
  if (length(C) != 2) {
    return(FALSE)
  }
  
  # Filter data to the selected categories and remove NAs
  selected_data <- data %>%
    filter(.data[[A]] %in% C) %>%
    select(all_of(c(A, B))) %>%
    na.omit()
  
  # Group by the categorical variable and calculate statistics
  group_stats <- selected_data %>%
    group_by(.data[[A]]) %>%
    summarise(
      `Number of Observations` = n(),
      Mean = round(mean(.data[[B]]), 3),
      `Standard Deviation` = round(sd(.data[[B]]), 3),
      Variance = round(var(.data[[B]]), 3)
    ) %>%
    rename(`Groups` = .data[[A]])
  
  variance_ratio <- max(group_stats$Variance) / min(group_stats$Variance)
  if (variance_ratio > 1.5) {
    return(FALSE)
  } else {
    return(TRUE)
  }
}

check_counts <- function(data, A, B, C){
  # Ensure exactly two categories are selected
  if (length(C) != 2) {
    return(FALSE)
  }
  
  # Filter data to the selected categories and remove NAs
  selected_data <- data %>%
    filter(.data[[A]] %in% C) %>%
    select(all_of(c(A, B))) %>%
    na.omit()
  
  # Group by the categorical variable and calculate statistics
  group_stats <- selected_data %>%
    group_by(.data[[A]]) %>%
    summarise(
      `Number of Observations` = n(),
      Mean = round(mean(.data[[B]]), 3),
      `Standard Deviation` = round(sd(.data[[B]]), 3),
      Variance = round(var(.data[[B]]), 3)
    ) %>%
    rename(`Groups` = .data[[A]])
  
  if (any(group_stats$`Number of Observations` < 30)) {
    return(FALSE)
  } else {
    return(TRUE)
  }
}


# Define server logic required to draw a histogram
server <- function(input, output, session) {
  # chi-sq test
  output$var1_levels <- renderUI({
    req(input$var1)
    var1_data <- na.omit(data[[input$var1]])  # Remove missing values
    var1_levels <- unique(var1_data)
    radioButtons("categorical_interest1", "Select categorical interest of the variable for the test:", choices = var1_levels, selected = var1_levels[1])
  })
  
  output$dist_plot1 <- plotly::renderPlotly({
    req(input$var1)  # Ensure inputs are available
    var_data <- na.omit(data[[input$var1]])  # Extract the data for the selected variable
    plot_ly(x = var_data, type = "histogram", marker = list(color = 'pink')) %>%
      layout(title = "Histogram of Variable 1",  # Add the title
             xaxis = list(title = "Categories of this Variable"),          # X-axis label
             yaxis = list(title = "Count of Observations"))             # Y-axis label
  })
  
  output$dist_plot2 <- plotly::renderPlotly({
    req(input$var2)  # Ensure inputs are available
    var_data <- na.omit(data[[input$var2]])  # Extract the data for the selected variable
    plot_ly(x = var_data, type = "histogram", marker = list(color = 'pink'))  %>%
      layout(title = "Histogram of Variable 2",  # Add the title
             xaxis = list(title = "Categories of this Variable"),          # X-axis label
             yaxis = list(title = "Count of Observations"))             # Y-axis label
  })
  
  output$des_p3 <- renderUI({
    req(input$var1, input$var2)
    
    # Dynamically generate the description based on user inputs
    description <- paste(
      "The interactive stacked histogram shows the distribution of", 
      "<strong>",
      input$var1, "</strong>",
      "across different levels of",
      "<strong>",
      input$var2, "</strong> <br/>",
      ".",
      "Each bar represents a category of", input$var1, 
      "with the height of the bars representing the frequency.",
      "The different colors within each bar correspond to the categories of", input$var2,
      "allowing you to visually compare the contribution of each category.",
      "<br>"
    )
    title_name <- paste("Distribution of ", "<strong>", input$var2, "</strong>",
                        " on ", "<strong>", input$var1, "</strong>")
    
    final <- paste(description, "<br><br>", title_name, "<br><br>")
    HTML(final)
  })
  
  output$dist_plot3 <- plotly::renderPlotly({
    req(input$var1, input$var2, input$categorical_interest1)
    
    # Calculate the contingency table
    var1_data <- na.omit(data[[input$var1]])
    var1_levels <- unique(var1_data)
    
    if (length(var1_levels) == 2) {
      contingency_table_result <- get_table(data, input$var1, input$var2, input$categorical_interest1)
    } else {
      contingency_table_result <- get_result(data, input$var1, input$var2, input$categorical_interest1)
    }
    
    # Convert the contingency table to a data frame suitable for ggplot
    data_mat1 <- as.data.frame.table(contingency_table_result)
    colnames(data_mat1) <- c("Variable_1", "Variable_2", "Frequency")  # Rename columns
    
    # Create the stacked histogram plot with ggplot2
    p1 <- data_mat1 %>%
      ggplot(aes(x = Variable_1, y = Frequency, fill = Variable_2)) +
      geom_col() +  # Create a bar plot
      scale_fill_brewer(palette = "Set6") +  # Use a palette for distinct colors
      labs(
        title = "Histogram of data distribution of the selected categories",  # Dynamically set the title
        x = input$var1,  # X-axis label
        y = "Frequency"  # Y-axis label
      ) +
      theme(
        plot.title = element_text(hjust = 0.5),  # Center the title
        axis.text.x = element_text(hjust = 1)  # Rotate x-axis labels
      )
    
    # Convert the ggplot to an interactive plotly plot
    tg1_plot <- ggplotly(p1)
    
    # Return the interactive plot
    tg1_plot
  })
  
  
  output$table_chi <- DT::renderDT({
    req(input$var1, input$var2, input$categorical_interest1)
    var1_data <- na.omit(data[[input$var1]])
    var1_levels <- unique(var1_data)

    if (length(var1_levels) == 2) {
      contingency_table_result <- get_table(data, input$var1, input$var2, input$categorical_interest1)
    } else {
      contingency_table_result <- get_result(data, input$var1, input$var2, input$categorical_interest1)
    }
    
    contingency_df <- as.data.frame.matrix(contingency_table_result)
    contingency_df <- tibble::rownames_to_column(contingency_df, var = "Categories")  # Add the row names as a column
    
    datatable(
      contingency_df,
      options = list(pageLength = 5),
      caption = "Contingency Table for selected category of interest and the variable 2"
    )
  })
  
  output$table_freq <- renderDataTable({
    req(input$var1, input$var2, input$categorical_interest1)
    var1_data <- na.omit(data[[input$var1]])
    var1_levels <- unique(var1_data)
    
    if (length(var1_levels) == 2) {
      contingency_table_result <- get_table(data, input$var1, input$var2, input$categorical_interest1)
    } else {
      contingency_table_result <- get_result(data, input$var1, input$var2, input$categorical_interest1)
    }
    
    test_result <- chisq.test(contingency_table_result)
    expected_val <- round(test_result$expected, 2)
    
    expected_df <- as.data.frame.matrix(expected_val)
    expected_df <- tibble::rownames_to_column(expected_df, var = "Categories")
    
    # Render the expected values table
    datatable(
      expected_df,
      options = list(pageLength = 5),
      caption = "Expected Frequencies for the Chi-square Test"
    )
  })
  
  output$assumption1 <- renderUI({
    req(input$var1, input$var2, input$categorical_interest1)
    
    text <- paste(
      "<span style='font-size: 20px;'>",
      "<strong> Assumptions check: </strong><br/>",
      "</span><br>",
      "<strong> 1: Does the sample make of iid observations?</strong><br/>",
      "This sample is not a random sample since it only collected responses from volunteering students who are willing to do the survey, resulting in non-response bias.",
      "For a data to be a random sample, each student in the population (assumed DATA2x02 in our case) should have the equal chance to be selected to answer the survey question which shouldn’t involve individual’s decision.",
      "However, due to the purpose of hypothesis testing in this study, it is assumed that the data is randomly sampled from the students. Thus, the data is considered iid.", 
      "The answer from each student is independent since the students are unable to see other students' answer so the answer of each student would not affect other students.",
      "<br><br>",
      "<strong> 2: Does the expected value greater than 5 under the null hypothesis? (The expected values are shown in the Frequency table above). </strong><br/>"
    )
    
    above_5 <- above_5_chi(data, input$var1, input$var2, input$categorical_interest1)
    second <- if (above_5){
      "Yes, all expected values aer greater than 5, assumptions achieved."
    } else {
      "No, not all expected values are greater than 5, assumptions not achieved. Please select new category of interest or new variables."
    }
    
    HTML(text, second, "<br><br><br>")
  })
  
  output$chiq_test <- renderUI({
    # Interpretation based on the test result
    req(input$var1, input$var2, input$categorical_interest1)
    var1_data <- na.omit(data[[input$var1]])
    var1_levels <- unique(var1_data)
    
    if (length(var1_levels) == 2) {
      contingency_table_result <- get_table(data, input$var1, input$var2, input$categorical_interest1)
    } else {
      contingency_table_result <- get_result(data, input$var1, input$var2, input$categorical_interest1)
    }
    test_result <- chisq.test(contingency_table_result)
    above_5 <- above_5_chi(data, input$var1, input$var2, input$categorical_interest1)
    
    if (input$chi_type == "Independence Test" && above_5){
      withMathJax(HTML(paste(
        "<strong>Chi-square Test Result (Independence):</strong> <br/>",
        "<br>",
        "Expected value: ", 
        "$$e_{ij} = n \\times \\hat{p_{ij}} = \\hat{n} \\times \\hat{p_i} \\hat{p_j} = \\frac{y_i y_j}{n}$$",
        "Test Statistic: ", 
        "$$ T = \\sum_{i=1}^r \\sum_{j=1}^c \\frac{(y_{ij} - e_{ij})^2}{e_{ij}} \\sim \\chi^2_{(r-1)(c-1)} \\text{Under } H_0$$",
        "<br/>",
        "Observed test statistic: ", round(test_result$statistic, 3), 
        "$$t_0 = \\sum_{i=1}^{r} \\sum_{j=1}^{c} \\frac{(y_{ij} - \\frac{y_i y_j}{n})^2}{\\frac{y_i y_j}{n}} \\sim \\chi^2_{(r-1)(c-1)}$$",
        "<br/>", 
        "Degrees of Freedom: ", test_result$parameter, "<br/>", "$$(r-1)(c-1)$$",
        "<br>",
        "P-value: ", round(test_result$p.value, 3), "<br/>",
        "$$P(T \\ge t_0) = P(\\chi_{(r-1)(c-1)}^2 \\ge t_0)$$",
        "Decision: ", "$$\\text{Reject } H_0 \\text{if the p value is less than 0.05}$$", 
        "$$\\text{Accept } H_0 \\text{if the p value is larger than 0.05}$$",
        "<br>"
      )))
    } else if (input$chi_type == "Homogeneity Test" && above_5){
      withMathJax(HTML(paste(
        "<strong>Chi-square Test Result (Homogeneity):</strong> <br/>",
        "<br>",
        "Expected value: ", 
        "$$e_{ij} = n_i \\times \\hat{p_{ij}} = \\frac{y_i y_j}{n} = \\frac{\\text{row i total} \\times \\text{row j total}}{\\text{Overall total}}$$",
        "Test Statistic: ", 
        "$$ T = \\sum_{i=1}^r \\sum_{j=1}^c \\frac{(y_{ij} - e_{ij})^2}{e_{ij}} \\sim \\chi^2_{(r-1)(c-1)} \\text{Under } H_0$$",
        "<br/>",
        "Observed test statistic: ", round(test_result$statistic, 3), 
        "$$t_0 = \\sum_{i=1}^{r} \\sum_{j=1}^{c} \\frac{(y_{ij} - e_{ij})^2}{e_{ij}} \\sim \\chi^2_{(r-1)(c-1)}$$",
        "<br/>", 
        "Degrees of Freedom: ", test_result$parameter, "<br/>", "$$(r-1)(c-1)$$",
        "<br>",
        "P-value: ", round(test_result$p.value, 3), "<br/>",
        "$$P(T \\ge t_0) = P(\\chi_{(r-1)(c-1)}^2 \\ge t_0)$$",
        "Decision: ", "$$\\text{Reject } H_0 \\text{if the p value is less than 0.05}$$", 
        "$$\\text{Accept } H_0 \\text{if the p value is larger than 0.05}$$",
        "<br>"
      )))
    }
    
  })
  
  output$chi_interpretation <- renderUI({
    req(input$var1, input$var2, input$categorical_interest1)
    var1_data <- na.omit(data[[input$var1]])
    var1_levels <- unique(var1_data)
   
    
    if (length(var1_levels) == 2) {
      contingency_table_result <- get_table(data, input$var1, input$var2, input$categorical_interest1)
    } else {
      contingency_table_result <- get_result(data, input$var1, input$var2, input$categorical_interest1)
    }
    test_result <- chisq.test(contingency_table_result)
    above_5 <- above_5_chi(data, input$var1, input$var2, input$categorical_interest1)
    
    if (input$chi_type == "Independence Test") {
      if (above_5){
        withMathJax(HTML(paste(
          "<strong> Assumptions achieved. </strong><br><br>",
          "<strong> Hypothesis of independence test: </strong><br><br>",
          "Null hypothesis: ", "$$H_0: \\text{The variables are independent to each other.}$$",
          "$$p_{ij} = p_i \\times p_j, i = \\text{# of rows}, j = \\text{# of columns}$$",
          "Alternative hypothesis: ", "$$H_a: \\text{The variables are dependent on each other.}$$", 
          "<br><br>"
        )))
      } else {
        HTML("<strong> Assumptions not achieved. Failed Assumptions. </strong><br>", 
             "There is expected value less than 5. Please reselect category of interest or reselect the variables.")
      }
      
    } else {
      if (above_5){
        withMathJax(HTML(paste(
          "<strong> Assumptions achieved. </strong><br><br>",
          "<strong> Hypothesis of homogeneity test: </strong><br><br>",
          "Null hypothesis: ", "$$H_0: \\text{The distribution of the categorical variable is the same (homogeneous)} \\\\
          \\text{for each subgroup or population.}$$",
          "$$p_{11} = p_{21} = ... = p_{r1} \\text{ and } p_{12} = p_{22} = ... = p_{r2}$$",
          "<br>", 
          "Alternative hypothesis: ", 
          "$$ H_a: \\text{The distribution of the categorical variable is different (nonhomogeneous)} \\\\
          \\text{for each subgroup or population (not all equalities hold).} $$",
          "<br><br>"
        )))
      } else {
        HTML("<strong> Assumptions not achieved. Failed Assumptions. </strong><br>", 
             "There is expected value less than 5. Please reselect category of interest or reselect the variables.")
      }
    }
  })
  
  output$chi_conclusion <- renderUI({
    req(input$var1, input$var2, input$categorical_interest1)
    var1_data <- na.omit(data[[input$var1]])
    var1_levels <- unique(var1_data)
    
    if (length(var1_levels) == 2) {
      contingency_table_result <- get_table(data, input$var1, input$var2, input$categorical_interest1)
    } else {
      contingency_table_result <- get_result(data, input$var1, input$var2, input$categorical_interest1)
    }
    test_result <- chisq.test(contingency_table_result)
    above_5 <- above_5_chi(data, input$var1, input$var2, input$categorical_interest1)
    
    conc <- if (input$chi_type == "Independence Test" && above_5){
      if (test_result$p.value < 0.05) {
        paste("<span style='font-size: 18px;'>", "<strong> Conclusion: </strong>", "</span><br><br>" ,
          "Since p value is less than 0.05, there is significant evidence to suggest that there is an association between the two variables.", 
        "Thus, the variables are dependent on each other.")
      } else {
        paste("<span style='font-size: 18px;'>", "<strong> Conclusion: </strong>", "</span><br><br>" ,
          "Since p value is greater than 0.05, there is not enough evidence to suggest that there is an association between the two variables.",
        "Thus, the variables are independent to each other.")
      }
    } else if (input$chi_type == "Homogeneity Test" && above_5) {
      if (test_result$p.value < 0.05) {
        paste("<span style='font-size: 18px;'>", "<strong> Conclusion: </strong>", "</span><br><br>" ,
          "Since p value is less than 0.05, ", 
              "there is sufficient evidence to suggest that the proportions across the groups differ.",
              "Thus, the distribution of the categorical variables is different for each subgroup or population, ",
              "and there is a significant difference in the proportion of the variable 2 categories between the variable 1 categories.")
      } else {
        paste("<span style='font-size: 18px;'>", "<strong> Conclusion: </strong>", "</span><br><br>" ,
          "Since p value is greater than 0.05, ", 
              "there is not enough evidence to suggest that the proportions across the groups differ.",
              "Thus, the distribution of the categorical variables is same for each subgroup or population, ",
              "and the proportion of the variable 2 categories between the variable 1 categories is homogeneous.")
      }
    }
    HTML(conc)
  })
  
  output$graph_assum <- renderUI({
    req(input$var1, input$var2, input$categorical_interest1)
    above_5 <- above_5_chi(data, input$var1, input$var2, input$categorical_interest1)
    if (above_5){
      HTML(paste("<span style='font-size: 18px;'>", 
           "Assumptions satisfied. The chi square distribution is shown below with the results from the chi sqaure test.", 
           "</span><br><br>"))
    } else {
      HTML(paste("<span style='font-size: 18px;'>", 
           "Assumptions not statisfied. The chi square distribution would not be shown.",
           "</span><br><br>"))
    }
  })
  
  output$graph_chi <- plotly::renderPlotly({
    req(input$var1, input$var2, input$categorical_interest1)
    var1_data <- na.omit(data[[input$var1]])
    var1_levels <- unique(var1_data)
    
    if (length(var1_levels) == 2) {
      contingency_table_result <- get_table(data, input$var1, input$var2, input$categorical_interest1)
    } else {
      contingency_table_result <- get_result(data, input$var1, input$var2, input$categorical_interest1)
    }
    test_result <- chisq.test(contingency_table_result)
    above_5 <- above_5_chi(data, input$var1, input$var2, input$categorical_interest1)
    
    test_stat <- test_result$statistic
    p_value <- test_result$p.value
    df <- test_result$parameter
    
    x_vals <- seq(0, test_stat + 4 * sqrt(2 * df), length.out = 1000)
    chi_square_density <- dchisq(x_vals, df = df)
    
    if (above_5){
      p <- ggplot(data = data.frame(x = x_vals, density = chi_square_density), aes(x = x, y = density)) +
        geom_line(color = "grey", size = 1) +
        geom_vline(xintercept = test_stat, color = "red", linetype = "dashed", size = 1) +
        annotate("text", 
                 x = test_stat + 1,  # Slightly offset the test statistic for better visibility
                 y = dchisq(test_stat, df) * 1.3,  # Place it slightly below the actual density value at the test statistic
                 label = paste("Test statistic =", round(test_stat, 3)),
                 color = "red", 
                 hjust = 0) +
        labs(title = paste("Chi Square distribution (Degree of Freedom =", df, ")"),
             x = "Chi Square Values",
             y = "Density") +
        theme_minimal()
      chi_square_value <- ifelse(x_vals > test_stat, x_vals, NA)
      
      # Highlight the p-value region
      p <- p + geom_area(data = data.frame(x = x_vals, density = chi_square_density),
                         aes(x = chi_square_value, y = density), fill = "pink", alpha = 0.3) +
        annotate("text", x = test_stat + 1, y = dchisq(test_stat, df) * 1.3 - 0.5,
                 label = paste("p-value =", round(p_value, 4)),
                 color = "steelblue", vjust = 1.5, hjust = 0)
      plotly::ggplotly(p)
    }
    
  })
  
  # t test
  output$cat1_levels <- renderUI({
    req(input$cat1)
    cat1_data <- na.omit(data[[input$cat1]])  # Remove missing values
    cat1_levels <- sort(unique(cat1_data))
    checkboxGroupInput("categorical_interest2", 
                       "Select up to two categorical interests of the variable to be the two samples 
                       (if more than two being selected, the first two categories in ascending 
                       alphabetical order would be applied):", 
                       choices = cat1_levels)
  })
  
  observeEvent(input$categorical_interest2, {
    # Get the currently selected values
    selected_vals <- input$categorical_interest2
    
    if (length(selected_vals) > 2) {
      updated_selections <- head(selected_vals, 2)
      updateCheckboxGroupInput(session, "categorical_interest2", 
                               selected = updated_selections)
    } 
  })
  
  output$boxplot_output <- plotly::renderPlotly({
    req(input$num2)  # Ensure the numeric input is available
    
    # Remove missing values from the numeric variable
    var2_data <- na.omit(data[[input$num2]])
    y_title <- input$num2
    
    # Split title into multiple lines if it's too long (example threshold: 20 characters)
    if (nchar(y_title) > 50) {
      y_title <- paste(strwrap(y_title, width = 50), collapse = "<br>")
    }
    
    # Create the plotly boxplot for the numeric variable
    plot_ly(x = "DATA2x02", y = var2_data, type = "box", boxpoints = "all", jitter = 0.3, 
            pointpos = -1.8, marker = list(opacity = 0.5)) %>%
      layout(title = "Boxplot of the numeric variable",
             yaxis = list(title = y_title),
             xaxis = list(title = ""))
  })
  
  output$hist_output <- plotly::renderPlotly({
    req(input$cat1)  # Ensure inputs are available
    var_data <- na.omit(data[[input$cat1]])  # Extract the data for the selected variable
    
    plot_ly(x = var_data, type = "histogram", marker = list(color = 'skyblue')) %>%
      layout(title = "Histogram of Categorical Variable",  # Add the title
             xaxis = list(title = "Categories"),          # X-axis label
             yaxis = list(title = "Count of Observations"))             # Y-axis label
  })
  
  output$warn_text <- renderUI({
    if (length(input$categorical_interest2) < 2) {
      HTML(paste(
        "<span style='font-size: 18px;'>",
        '<p style="color:blue;">', 
        "WARNING: Please select two categories from the selected Variable 1.",
        '</p>',
        "</span><br>"))
    } else {
      HTML(paste(
        "<span style='font-size: 18px;'>",
        "Categories selected: ",
        input$categorical_interest2[1], 
        " and ",
        input$categorical_interest2[2],
        "<br><br>",
        "Categorical Variable: ", input$cat1,
        "<br>",
        "Numerical Variable: ", input$num2,
        "</span><br>"
      ))
    }
  })
  
  output$side_by_side <- plotly::renderPlotly({
    req(input$cat1, input$num2)  # Ensure inputs are available
    
    cleaned_data <- data[!is.na(data[[input$cat1]]) & !is.na(data[[input$num2]]), ]
    
    selected_categories <- input$categorical_interest2
   
    if (length(selected_categories) > 2) {
      selected_categories <- head(sort(selected_categories), 2)
    } 
    
    filtered_data <- cleaned_data[cleaned_data[[input$cat1]] %in% head(sort(selected_categories), 2), ]
    
    # Create a data frame for plotting
    plot_data <- data.frame(Category = filtered_data[[input$cat1]], 
                            Value = filtered_data[[input$num2]])
    
    y_title <- input$num2
    if (nchar(y_title) > 50) {
      y_title <- paste(strwrap(y_title, width = 50), collapse = "<br>")
    }
    
    x_title <- input$cat1
    if (nchar(x_title) > 80) {
      x_title <- paste(strwrap(x_title, width = 80), collapse = "<br>")
    }
    
    plot_ly(data = plot_data, y = ~Value, type = "box", 
           color = ~Category, boxpoints = "all", jitter = 0.3, 
           pointpos = -1.8, marker = list(opacity = 0.5)) %>%
      layout(title = "Boxplot of the numeric variable by the categorical varibale",
             xaxis = list(title = x_title),
             yaxis = list(title = y_title))
  })
  
  output$t_assum <- renderUI({
    assum <- paste(
      "<strong> Assumptions: </strong><br><br>",
      "The data observations of both samples should be iid random variables following the normal distribution approximately with the same variance.",
      "$$X_1, ..., X_{n_x} \\text{ are iid } N(\\mu_x, \\sigma^2), Y_1, ..., Y_{n_y} \\text{ are iid } N(\\mu_y, \\sigma^2) \\text{ and } X_i'\\text{s are independent of } Y_i'\\text{s} $$", 
      "<br>",
      "The observations of both samples are considered as iid and randomly sampled since each student answer the question independently and each student' responses would not affect other students."
    )
    withMathJax(HTML(assum))
  })
  
  output$desc_qq <- renderUI({
    req(input$cat1, input$num2, input$categorical_interest2)
    var_check <- check_variance(data, input$cat1, input$num2, input$categorical_interest2)
    count_check <- check_counts(data, input$cat1, input$num2, input$categorical_interest2)
    
    assump2 <- paste(
      "<strong> Assumptions checking: </strong><br><br>",
      "<strong> 1: Following normal distribution or not </strong><br>",
      "According to the qq plot shown above and since we are performing a t test: ",
      "If most of the data points are close to the line which is the expected values for sampling from a normal distribution, the observations are following the normal distribution.",
      "If the data points do not perform a close fit to the line, but the sample size of both samples is greater than 30, the distribution of the test statistic is likely to follow t-distribution.",
      "This suggests that the inferences from the samples would be considered as valid.",
      "<br><br>",
      "<strong>2: Same variance for both samples: </strong><br>",
      "If the sample variance is significantly different, Welch two sample t-test might be a better option.",
      "The variance is check by dividing the larger variance by the smaller variance after comparing the variance value of the two groups.",
      '<p style="color:red;">', 
      "<br> <strong>If the variance ratio obtained from the division is greater than 1.5 would be considered as significant difference between the sample variance.</strong>",
      '</p>',
      "<br>"
    )
    
    meet <- if(var_check && count_check){
      paste("<span style='font-size: 16px;'>", 
            "Both assumptions are satisfied!", 
            "</span><br>",
            "The variances between the two groups are approximately equal for the assumption of equal variances. Two sample t-test is good to be performed! <br>") 
      } else {
        if (var_check) {
          paste(
            "<span style='font-size: 16px;'>", "Failed assumption 1", "</span><br>",
            "The variances between the two groups are approximately equal for the assumption of equal variances. Two sample t-test is good to be performed if assumption 1 statisfied. <br>",
            "The sample size is not large enough. Probably change your selection of groups or variables."
        )
      } else {paste(
        "<span style='font-size: 16px;'>", "Failed assumption 2", "</span><br>",
        "The variances between the two groups are too different for the assumption of equal variances. Welch two sample t-test would be better.",
        "The sample size is large enough to satisfied CLT, so valid interference from the data and thus satisfied assumption 1."
        )
      }
    }
    
    HTML(paste(assump2, "<strong> Satisfied or not: </strong><br>", meet))
  })
  
  
  output$var_output <- renderUI({
    req(input$cat1, input$num2, input$categorical_interest2)
    
    ini_des <- paste(
      "<strong> Categorical Variable: </strong><br>",
      input$cat1,
      "<br><br>",
      "<strong> Numerical Variable: </strong><br>", input$num2,
      "<br><br>",
      "<strong> Categories selected: </strong><br>", 
      "Group 1 (x): ", input$categorical_interest2[1], 
      "<br>", "Group 2 (y):",
      input$categorical_interest2[2],
      "<br><br>"
    )
    HTML(ini_des)
  })
  
  output$t_test_result <- renderPrint({
    req(input$cat1, input$num2, input$categorical_interest2, input$t_type)
    var_check <- check_variance(data, input$cat1, input$num2, input$categorical_interest2)
    count_check <- check_counts(data, input$cat1, input$num2, input$categorical_interest2)
    
    if (var_check){
      t_test_result <- get_t_result(data, input$cat1, input$num2, input$categorical_interest2, input$t_type)
    } else {
      t_test_result <- get_welch_result(data, input$cat1, input$num2, input$categorical_interest2, input$t_type)
    }
    
    # Output the t-test result
    print(t_test_result)
  })
  
  output$t_test <- renderUI({
    req(input$cat1, input$num2, input$categorical_interest2, input$t_type)
    
    var_check <- check_variance(data, input$cat1, input$num2, input$categorical_interest2)
    count_check <- check_counts(data, input$cat1, input$num2, input$categorical_interest2)
    
    if (var_check){
      t_test_result <- get_t_result(data, input$cat1, input$num2, input$categorical_interest2, input$t_type)
    } else {
      t_test_result <- get_welch_result(data, input$cat1, input$num2, input$categorical_interest2, input$t_type)
    }
    
    hypo <- paste(
      "<strong> Hypothesis of two sample t test: </strong><br/><br>",
      "Null hypothese: The two groups (", input$categorical_interest2[1], 
      " and ", input$categorical_interest2[2], 
      ") have the same mean in the population. <br>",
      "$$H_0: \\mu_x = \\mu_y$$", 
      "<br>"
    )
    
    alter <- if (input$t_type == "Greater"){paste(
      "Alternative hypothese: The first group (", input$categorical_interest2[1],
      ") has greater mean than the second group (", input$categorical_interest2[2], ") <br>",
      "$$H_a: \\mu_x \\gt \\mu_y$$")
    } else if (input$t_type == "Less"){paste(
      "Alternative hypothese: The first group (", input$categorical_interest2[1],
      ") has smaller mean than the second group (", input$categorical_interest2[2], ") <br>",
      "$$H_a: \\mu_x \\lt \\mu_y$$")
    } else if (input$t_type == "Two Sided") {paste(
      "Alternative hypothese: The mean of the first group (", input$categorical_interest2[1],
      ") is not equal to the second group (", input$categorical_interest2[2], ") <br>",
      "$$H_a: \\mu_x \\ne \\mu_y$$")
    }
    
    if (!count_check) {
      asum <- "<span style='font-size: 15px;'> WARNING: There are less than 30 observations in at least one of the selected groups, failed CLT, likely to fail assumption 1. <br> Better to select other groups or other variables to perform the t-test. </span> <br><br>"
    } else {
      asum <- "<span style='font-size: 15px;'> Satisfied assumptions! </span><br><br>"
    }
    
    if (length(input$categorical_interest2) == 2){
      rest <- paste(
        "<strong> Test Statistic: </strong>",
        "$$T = \\frac{\\bar{X} - \\bar{Y}}{S_p \\sqrt{\\frac{1}{n_x} + \\frac{1}{n_y}}}, \\text{ where } S_p^2 = \\frac{(n_x-1)S_x^2 + (n_y-1)S_y^2}{n_x + n_y - 2} \\text{. Under } H_0, T \\sim t_{n_x+n_y-2}.$$",
        "<br>",
        "<strong> Observed Test Statistic: </strong>", round(t_test_result$statistic, 3),
        "$$t_0 = \\frac{\\bar{x} - \\bar{y}}{s_p \\sqrt{\\frac{1}{n_x} + \\frac{1}{n_y}}}, \\text{ where } S_p^2 = \\frac{(n_x-1)s_x^2 + (n_y-1)s_y^2}{n_x + n_y - 2}$$",
        "<strong> Degree of freedom: </strong>", t_test_result$parameter,
        "$$n_x + n_y - 2$$",
        "<strong> p value: </strong>", round(t_test_result$p.value, 4)
        )
      
      p_value_for <- if (input$t_type == "Greater") {
        "$$P(t_{n_x+n_y-2} \\geq t_0)$$"
        
      } else if (input$t_type == "Less") {
        "$$P(t_{n_x+n_y-2} \\leq t_0)$$"
        
      } else if (input$t_type == "Two Sided"){
        "$$2P(|t_{n_x+n_y-2}| \\geq |t_0|)$$"
      }
      
    } else {
      rest <- paste(
        "<span style='font-size: 16px;'>",
        '<p style="color:blue;">', 
        "<br> WARNING: Please select exactly two categories to obtain the results.",
        '</p>',
        "</span><br>")
      p_value_for <- "<br>"
    }
    
    decision <- paste(
      "<strong> Decision: </strong><br><br>",
      "If the p value is less than 0.05, there is sufficient evidence to reject the null hypothesis. ",
      "<br><br>",
      "If the p value is greater than 0.05, the null hypothesis would be accepted since there is not enough evidence to reject it."
    )
    
    
    withMathJax(HTML(paste(asum, hypo, alter, rest, p_value_for, decision)))
  })
  output$norm_qq <- plotly::renderPlotly({
    req(input$num2, input$cat1, input$categorical_interest2)
    
    # Check if exactly two categories are selected
    if (length(input$categorical_interest2) != 2) {
      p <- ggplot() + 
        annotate("text", x = 0.5, y = 0.5, label = "Please select exactly two groups!") + 
        theme_void()
      ggplotly(p)
      
    } else {
      # Convert data columns and inputs to character to ensure proper comparison
      data[[input$cat1]] <- as.character(data[[input$cat1]])
      input_categories <- as.character(input$categorical_interest2)
      
      # Check if the numeric variable is actually numeric
      if (!is.numeric(data[[input$num2]])) {
        p <- ggplot() + 
          annotate("text", x = 0.5, y = 0.5, label = "The selected numeric variable is not numeric.") + 
          theme_void()
        ggplotly(p)
        
      } else {
        # Filter data to selected categories and remove NAs
        selected_data <- data %>%
          filter(.data[[input$cat1]] %in% input_categories) %>%
          select(all_of(c(input$cat1, input$num2))) %>%
          na.omit()
        
        if (nrow(selected_data) == 0) {
          p <- ggplot() + 
            annotate("text", x = 0.5, y = 0.5, label = "No data available for the selected groups.") + 
            theme_void()
          ggplotly(p)
          
        } else {
          # Ensure that both categories are present in selected_data
          present_categories <- unique(selected_data[[input$cat1]])
          if (!all(input_categories %in% present_categories)) {
            p <- ggplot() + 
              annotate("text", x = 0.5, y = 0.5, label = "Selected categories not present in data after removing missing values.") + 
              theme_void()
            ggplotly(p)
            
          } else {
            # Split data into two groups
            group_1 <- selected_data %>%
              filter(.data[[input$cat1]] == input_categories[1])
            group_2 <- selected_data %>%
              filter(.data[[input$cat1]] == input_categories[2])
            group_1_data <- group_1[[input$num2]]
            group_2_data <- group_2[[input$num2]]
            
            df_group_1 <- data.frame(value = group_1_data)
            df_group_2 <- data.frame(value = group_2_data)
            
            if (nrow(group_1) > 0 && nrow(group_2) > 0) {
              # Remove titles from individual plots
              p1 <- ggplot(df_group_1, aes(sample = value)) +
                stat_qq() +
                stat_qq_line() +
                labs(
                  x = "Theoretical Quantiles",
                  y = "Sample Quantiles"
                ) +
                theme_minimal()
              
              p2 <- ggplot(df_group_2, aes(sample = value)) +
                stat_qq() +
                stat_qq_line() +
                labs(
                  x = "Theoretical Quantiles",
                  y = "Sample Quantiles"
                ) +
                theme_minimal()
              
              # Convert the ggplot objects to plotly objects and adjust margins
              plotly_p1 <- ggplotly(p1) %>%
                layout(margin = list(t = 50))  # Increase top margin
              
              plotly_p2 <- ggplotly(p2) %>%
                layout(margin = list(t = 50))  # Increase top margin
              
              # Create the subplot
              side_plots <- subplot(
                plotly_p1, plotly_p2, nrows = 1, shareX = FALSE, shareY = FALSE,
                titleX = TRUE, titleY = TRUE, margin = 0.05
              )
              
              # Add annotations for the titles
              side_plots <- side_plots %>%
                layout(annotations = list(
                  list(
                    x = 0.15, y = 1.05, text = paste("QQ Plot for", input_categories[1]), 
                    showarrow = FALSE, xref='paper', yref='paper', 
                    xanchor='center', yanchor='bottom', font=list(size=16)
                  ),
                  list(
                    x = 0.75, y = 1.05, text = paste("QQ Plot for", input_categories[2]), 
                    showarrow = FALSE, xref='paper', yref='paper', 
                    xanchor='center', yanchor='bottom', font=list(size=16)
                  )
                ))
              
              side_plots  
              
            } else {
              p <- ggplot() + 
                annotate("text", x = 0.5, y = 0.5, label = "Not enough data for one or both QQ plots") + 
                theme_void()
              ggplotly(p)
            }
          }
        }
      }
    }
  })
  
  output$stats_table <- renderDataTable({
    req(input$cat1, input$num2, input$categorical_interest2)
    
    # Ensure exactly two categories are selected
    if (length(input$categorical_interest2) != 2) {
      return(data.frame(Message = "Please select exactly two categories to perform the t-test."))
    }
    
    # Filter data to the selected categories and remove NAs
    selected_data <- data %>%
      filter(.data[[input$cat1]] %in% input$categorical_interest2) %>%
      select(all_of(c(input$cat1, input$num2))) %>%
      na.omit()
    
    # Group by the categorical variable and calculate statistics
    group_stats <- selected_data %>%
      group_by(.data[[input$cat1]]) %>%
      summarise(
        `Number of Observations` = n(),
        Mean = round(mean(.data[[input$num2]]), 3),
        `Standard Deviation` = round(sd(.data[[input$num2]]), 3),
        Variance = round(var(.data[[input$num2]]), 3)
      ) %>%
      rename(`Groups` = .data[[input$cat1]])
    
    # Display the table
    datatable(
      group_stats, 
      options = list(pageLength = 5),
      caption = htmltools::tags$caption(
        style = 'caption-side: top; text-align: left; font-size: 18px; color: black;',
        paste('Summary Table: Group Statistics for the Selected Groups which are ', 
              input$categorical_interest2[1], "and", input$categorical_interest2[2])
      )
    )
  })
  
  output$t_conc <- renderUI({
    req(input$cat1, input$num2, input$categorical_interest2, input$t_type)
    var_check <- check_variance(data, input$cat1, input$num2, input$categorical_interest2)
    count_check <- check_counts(data, input$cat1, input$num2, input$categorical_interest2)
    
    if (var_check){
      t_test_result <- get_t_result(data, input$cat1, input$num2, input$categorical_interest2, input$t_type)
    } else {
      t_test_result <- get_welch_result(data, input$cat1, input$num2, input$categorical_interest2, input$t_type)
    }
    
    conc1 <- if (t_test_result$p.value > 0.05){
      paste("Since the p value is greater than 0.05, there is no enough evidence to reject the null hypothesis.",
            "The null hypothesis would be accepted, so that the two selected groups are likely to have the same mean value in population.")
    } else {
      paste("Since the p value is less than 0.05, the null hypothesis would be rejected",
            " and the alternative hypothesis would be accepted.")
    }
    
    conc2 <- if (t_test_result$p.value > 0.05){
      "<br><br>"
    } else {
      if (input$t_type == "Greater"){
        paste("The population mean of ", input$categorical_interest2[1], "is likely to be greater than ",
              input$categorical_interest2[2], ". <br>")
      } else if (input$t_type == "Less"){
        paste("The population mean of ", input$categorical_interest2[1], "is likely to be less than ",
              input$categorical_interest2[2], ". <br>")
      } else if (input$t_type == "two sided"){
        paste("The population mean of ", input$categorical_interest2[1], "is likely to be not equal to the population mean of ",
              input$categorical_interest2[2], ". <br>")
      }
    }
    
    HTML("<strong> Conclusion: </strong><br>", conc1, conc2)
  })
  
  output$graph_t <- renderPlot({
    req(input$cat1, input$num2, input$categorical_interest2, input$t_type)
    
    var_check <- check_variance(data, input$cat1, input$num2, input$categorical_interest2)
    count_check <- check_counts(data, input$cat1, input$num2, input$categorical_interest2)
    
    if (var_check){
      t_test_result <- get_t_result(data, input$cat1, input$num2, input$categorical_interest2, input$t_type)
    } else {
      t_test_result <- get_welch_result(data, input$cat1, input$num2, input$categorical_interest2, input$t_type)
    }
    
    if (length(input$categorical_interest2) == 2) {
      stat <- round(as.numeric(t_test_result$statistic), 3)
      p_value <- round(t_test_result$p.value,4)
      df <- as.numeric(t_test_result$parameter)
      
      alter <- input$t_type
      
      
      x_vals <- seq(-4, 4, length.out = 1000)
      t_density <- dt(x_vals, df = df)
      
      p <- ggplot(data = data.frame(x = x_vals, density = t_density), aes(x = x, y = density)) +
        geom_line(color = "lightblue", size = 1) +
        geom_vline(xintercept = stat, color = "red", linetype = "dashed", size = 1) +
        annotate("text", x = stat + 0.5, y = max(t_density) * 0.5 , 
                 label = paste("Test Statistic =", round(stat, 2)), 
                 color = "red", hjust = -0.1) +
        labs(title = paste("T Distribution (degree of freedom =", df, ")"),
             x = "t", 
             y = "") +
        theme_minimal()
      
      p <- p + annotate("text", x = stat+1, y = 0, 
                        label = paste("P-value =", round(p_value, 4)), 
                        color = "steelblue", vjust = 1.5, hjust = 0)
      print(p)
      #plotly::ggplotly(p)
    }
  })
  
}



# Run the application 
shinyApp(ui = ui, server = server)
