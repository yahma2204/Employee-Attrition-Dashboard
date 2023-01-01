# load required packages
library(rsample)   # data splitting
library(ggplot2)   # allows extension of visualizations
library(scales)
library(ggthemes)
library(dplyr)     # basic data transformation
library(stringr)
library(tibble)
library(iml)       # ML interprtation
library(glue)
library(shiny)
library(shinythemes)
library(shinydashboard)
library(shinyWidgets)
library(shinyjs)
library(DT)
library(plotly)
library(modeldata)
library(h2o)       # machine learning modeling

# initialize h2o session
h2o.init()

# classification data
data("attrition", package = "modeldata")
df <- attrition %>% 
  mutate_if(is.ordered, factor, ordered = FALSE) %>% 
  mutate(across(c('BusinessTravel','Department', 'Education',
                  'EducationField','EnvironmentSatisfaction',
                  'JobInvolvement','JobRole','JobSatisfaction',
                  'RelationshipSatisfaction'), str_replace, '_', ' ')) %>% 
  mutate_if(is.character, factor, ordered = FALSE) %>% 
  select(Age,BusinessTravel,DistanceFromHome,Education,EducationField,
         EnvironmentSatisfaction,Gender,JobInvolvement,JobLevel,
         JobRole,JobSatisfaction,MaritalStatus,NumCompaniesWorked,
         OverTime,RelationshipSatisfaction,TotalWorkingYears,
         TrainingTimesLastYear,WorkLifeBalance,YearsSinceLastPromotion,
         YearsWithCurrManager, Attrition)
df$ID_Employee <- rownames(attrition)

# convert to h2o object
df.h2o <- h2o::as.h2o(df)

# create train, validation, and test splits
split_h2o <- h2o::h2o.splitFrame(df.h2o, c(0.8, 0.10), seed = 1234 )
train_h2o <- h2o::h2o.assign(split_h2o[[1]], "train" ) # 80%
valid_h2o <- h2o::h2o.assign(split_h2o[[2]], "valid" ) # 10%
test_h2o  <- h2o::h2o.assign(split_h2o[[3]], "test" )  # 10%

# variable names for resonse & features
z <- "ID_Employee"
y <- "Attrition"
x <- dplyr::setdiff(names(df), c(y,z)) 

# Neural Network (30-21-16-3-2)
dl1 <- h2o.deeplearning(x = x,
                        y = y,
                        hidden = c(21,16,3),
                        epochs = 800, activation = "RectifierWithDropout",
                        training_frame = train_h2o,
                        validation_frame = valid_h2o, seed = 234)

# Predict on hold-out set, test_h2o
pred_h2o_dl1 <- h2o::h2o.predict(object = dl1, newdata = test_h2o)
test_performance <- test_h2o %>%
  tibble::as_tibble() %>%
  select(-Attrition) %>% 
  tibble::add_column(Attrition = as.vector(pred_h2o_dl1$predict)) %>%
  dplyr::mutate_if(is.character, as.factor) %>% 
  tibble::as_tibble()

# Run lime() on training set
explainer <- lime::lime(
  as.data.frame(train_h2o[,c("Age","BusinessTravel","DistanceFromHome","Education","EducationField",
                             "EnvironmentSatisfaction","Gender","JobInvolvement","JobLevel",
                             "JobRole","JobSatisfaction","MaritalStatus","NumCompaniesWorked",
                             "OverTime","RelationshipSatisfaction","TotalWorkingYears",
                             "TrainingTimesLastYear","WorkLifeBalance","YearsSinceLastPromotion",
                             "YearsWithCurrManager")]), 
  model          = dl1, 
  bin_continuous = FALSE)


# Define UI for application that draws a histogram
ui <- shinydashboard::dashboardPage(title = "Employee Attrition Dashboard",
  dashboardHeader(
    title=div(
      id = "img-id",
      img(src="blue_unpad_tsdn.jpg",
          height = 35,
      ), "TSDN 2022"
    ) 
  ),
  shinydashboard::dashboardSidebar(
    sidebarMenu(
      menuItem("Employee Analysis", icon = icon("dashboard"), tabName = "employee_analysis", badgeColor = "green"),
      menuItem("Input Data", tabName = "prediction", icon = icon("th")),
      menuItem("About", tabName = "about", icon = icon("address-card"))
    )
  ),
  shinydashboard::dashboardBody(
    tags$head(tags$style(HTML('
        .skin-blue .main-header .logo {
          background-color: #3c8dbc;
        }
        .skin-blue .main-header .logo:hover {
          background-color: #3c8dbc;
        }
        .small-box {height: 110px}
      '))),
    
    tabItems(
      tabItem(tabName = "employee_analysis",
              h2("Employee Analysis", style="text-align: center;"),
              fluidRow(
                shinydashboard::box(width = 3,
                    selectizeInput(inputId = "no_employee",
                                   label = "Select ID Employee:",
                                   choices = unique(test_performance$ID_Employee))
                    ),
                valueBoxOutput("No_employee", width = 3),
                valueBoxOutput("Label", width = 3),
                valueBoxOutput("Probability", width = 3),
                fluidRow(
                  shinydashboard::box(width = 12,
                      plotlyOutput("plot_indiv")))
              )),
      tabItem(tabName = "prediction",
              fluidRow(
                box(width = 6, height = 150,
                  fileInput("file_att", "Upload CSV File",
                            multiple = FALSE,
                            accept = c("text/csv",
                                       "text/comma-separated-values,text/plain",
                                       ".csv")),
                  actionButton("example", "Format Data")#,
                  # actionButton("reset", "Reset")
                  ),
                box(width = 6,height = 150,
                  selectizeInput("att_test_id",
                                 label = "Select ID Employee:", 
                                 choices = NULL, multiple = F#, width = 100000000
              ),
              actionButton("go", "Predict"))),
              fluidRow(
              valueBoxOutput("No_employee_test", width = 4),
              valueBoxOutput("Label_test", width = 4),
              valueBoxOutput("Probability_test", width = 4)),
              fluidRow(
                box(width = 12,
                  plotlyOutput("plot_att"))
             )),
      tabItem(tabName = "about",
              h2("Background"),
              p("Employees are the most valuable asset of a company. They add value to a company both in terms of quality and quantity. 
                However, retaining employees is a huge challenge for companies and can be quite costly. Employee turnover often occurs 
                in a company. Employee turnover or attrition can have a negative impact on the company and the employees working in the 
                company.", style="text-align: justify;"),
              p("In this study, we conducted an analysis of employee attrition. This study used several variables related to employee attrition. 
                These variables analyzed using several classification methods. The main objective of this research is to find out whether the 
                employee has a tendency to leave the company or not according to the variables in the employee attrition data. The result of 
                this study can be used by companies to reduce losses that occur due to employee attrition.", style="text-align: justify;")
              
              
              )
      )))

# Define server logic required to draw a histogram
server <- function(input, output, session) {
  
  output$No_employee <- renderValueBox({
      
      valueBox(
        value = input$no_employee,
        subtitle = "ID Employee",
        icon = icon("person"),
        color = "orange"
      )
      
    })
    
    output$Label <- renderValueBox({
      
      em <- test_performance %>% 
        as.data.frame() %>% 
        filter(ID_Employee %in% input$no_employee) %>% 
        select(Age,BusinessTravel,DistanceFromHome,Education,EducationField,
               EnvironmentSatisfaction,Gender,JobInvolvement,JobLevel,
               JobRole,JobSatisfaction,MaritalStatus,NumCompaniesWorked,
               OverTime,RelationshipSatisfaction,TotalWorkingYears,
               TrainingTimesLastYear,WorkLifeBalance,YearsSinceLastPromotion,
               YearsWithCurrManager) 
      
      explanation <- lime::explain(em, 
        explainer    = explainer, n_labels = 1,
        n_features   = 10,
        kernel_width = 0.5)
      
      valueBox(
        value = unique(explanation$label),
        subtitle = "Prediction",
        icon = if (unique(explanation$label) == "Yes") icon("check") else icon("xmark"),
        color = if (unique(explanation$label) == "Yes") "red" else "green"
      )
      
    })
    
    output$Probability <- renderValueBox({
      
      em <- test_performance %>% 
        as.data.frame() %>% 
        filter(ID_Employee %in% input$no_employee) %>% 
        select(Age,BusinessTravel,DistanceFromHome,Education,EducationField,
               EnvironmentSatisfaction,Gender,JobInvolvement,JobLevel,
               JobRole,JobSatisfaction,MaritalStatus,NumCompaniesWorked,
               OverTime,RelationshipSatisfaction,TotalWorkingYears,
               TrainingTimesLastYear,WorkLifeBalance,YearsSinceLastPromotion,
               YearsWithCurrManager) 
      
      explanation <- lime::explain(em, 
        explainer    = explainer,labels = "Yes",
        n_features   = 10,
        kernel_width = 0.5)
      
      valueBox(
        value = paste0(unique(round(explanation$label_prob,4))*100,"%"),
        subtitle = "Employee Attrition Probability",
        icon = icon("star"),
        color = "fuchsia"
      )
      
    })
    
    output$plot_indiv <- renderPlotly({
      
      em <- test_performance %>% 
        as.data.frame() %>% 
        filter(ID_Employee %in% input$no_employee) %>% 
        select(Age,BusinessTravel,DistanceFromHome,Education,EducationField,
               EnvironmentSatisfaction,Gender,JobInvolvement,JobLevel,
               JobRole,JobSatisfaction,MaritalStatus,NumCompaniesWorked,
               OverTime,RelationshipSatisfaction,TotalWorkingYears,
               TrainingTimesLastYear,WorkLifeBalance,YearsSinceLastPromotion,
               YearsWithCurrManager) %>%
        as.data.frame()
      
      # Run explain() on explainer
      explanation <- lime::explain(em, 
        explainer    = explainer, labels = "Yes",
        n_features   = 10,
        kernel_width = 0.5)
      
      explain_dl1 <- data.frame(Feature = explanation$feature_desc, 
                                Weight = explanation$feature_weight,
                                Label = glue('Weight : {round(explanation$feature_weight, 3)}'))
      
      plot_individual <- ggplot(mapping = aes(y = reorder(Feature, Weight), 
                                              x = Weight,  
                                              text = Label), data = explain_dl1) +
        geom_col(aes(fill=ifelse(Weight <0,"Contradict", "Support")))+
        scale_fill_manual(#name = "Weight",#guide=FALSE,
                          #labels = c("Support"="TRUE","Contradict"="FALSE"),
                          values = c("Support"="#ce4257","Contradict"="#0077b6"))+
        labs(x = "Weight", y = NaN, 
             title = "Employee Attrition Prediction",
             fill = "Status")+
        theme(legend.position = "left")
      
      plotly_individual <- ggplotly(plot_individual, tooltip = "text") %>% 
        layout(title = list(text = paste0('Employee Attrition Prediction')),
               margin = list(l = 0, r = 0,b = 5, t = 30,pad = 0),
               legend = list(orientation = "h"#, x = -0.5, 
                             ,y =-0.2
                             ))
      
      plotly_individual
      
    })
    
    observeEvent(input$example,{
      showModal(modalDialog(
        title = h3("Format Data", style="font-weight: bold;"),
        h4("Variable required :"),
        tags$ol(
          tags$li("Age"), 
          tags$li("BusinessTravel : Non-Travel, Travel Frequently, Travel Rarely"), 
          tags$li("DistanceFromHome"),
          tags$li("Education : Bachelor, Below College, College, Doctor, Master"), 
          tags$li("EducationField : Human Resources, Life Sciences, Marketing, Medical, Other, Technical Degree"), 
          tags$li("EnvironmentSatisfaction : Low, Medium, High, Very High"),
          tags$li("Gender : Female, Male"), 
          tags$li("JobInvolvement : Low, Medium, High, Very High"), 
          tags$li("JobLevel"),
          tags$li("JobRole : Sales Executive, Sales Representative, Research Director, Research Scientist, 
                  Laboratory Technician, Manufacturing Director, Healthcare Representative, Manager, Humas Resources"), 
          tags$li("JobSatisfaction : Low, Medium, High, Very High"), 
          tags$li("MaritalStatus : Divorced, Married, Single"),
          tags$li("NumCompaniesWorked"), 
          tags$li("OverTime : No, Yes"), 
          tags$li("RelationshipSatisfaction"),
          tags$li("TotalWorkingYears"),
          tags$li("TrainingTimesLastYear"), 
          tags$li("WorkLifeBalance : Bad, Best, Better, Good"), 
          tags$li("YearsSinceLastPromotion"),
          tags$li("NumCompaniesWorked"), 
          tags$li("OverTime : No, Yes"), 
          tags$li("YearsWithCurrManager"),
          tags$li("ID_Employee"), style="text-align: justify;"),
      
        p("Here is the example of the data:",tags$a(href="https://drive.google.com/uc?export=download&id=1msEbDM13j2SXck0cEM7dIGqii-5iuLZk", strong("(Click here)"), style = 'color:black;'))
      ))
    })
    
    randomVals <- eventReactive(input$go, {
      print(input$att_test_id)
    })
    
    rv <- reactiveValues(data = NULL)
    
    # Update Dataset
    reactive_att<-reactive({
      req(input$file_att)
      rv$data <- read.csv(input$file_att$datapath)
      data_test_att <- rv$data
      return(data_test_att)
    })
    
    observeEvent(input$reset, {
      rv$data <- NULL
    })
    
    observe({
      data_test_att <- reactive_att()
      updateSelectizeInput(session, inputId = "att_test_id",
                           label = "Select ID Employee:",
                           choices = unique(data_test_att$ID_Employee),
                           selected = head(unique(data_test_att$ID_Employee),1)
                           )
    })
    
   
    output$No_employee_test <- renderValueBox({
      
      data_test_att <- reactive_att()
      rownames(data_test_att) <- data_test_att$ID_Employee
      em <- data_test_att %>% 
        # t() %>% 
        # as_tibble() %>% 
        mutate_if(is.character, factor, ordered = FALSE) %>%
        filter(ID_Employee %in% randomVals()) %>%
        select(Age,BusinessTravel,DistanceFromHome,Education,EducationField,
               EnvironmentSatisfaction,Gender,JobInvolvement,JobLevel,
               JobRole,JobSatisfaction,MaritalStatus,NumCompaniesWorked,
               OverTime,RelationshipSatisfaction,TotalWorkingYears,
               TrainingTimesLastYear,WorkLifeBalance,YearsSinceLastPromotion,
               YearsWithCurrManager) #%>%
      # as_tibble()
      explainer <- lime::lime(
        as.data.frame(train_h2o[,c("Age","BusinessTravel","DistanceFromHome","Education","EducationField",
                                   "EnvironmentSatisfaction","Gender","JobInvolvement","JobLevel",
                                   "JobRole","JobSatisfaction","MaritalStatus","NumCompaniesWorked",
                                   "OverTime","RelationshipSatisfaction","TotalWorkingYears",
                                   "TrainingTimesLastYear","WorkLifeBalance","YearsSinceLastPromotion",
                                   "YearsWithCurrManager")]), 
        model          = dl1, 
        bin_continuous = FALSE)
      
      # Run explain() on explainer
      explanation <- lime::explain(em, 
                                   explainer    = explainer, labels = "Yes",
                                   n_features   = 10,
                                   kernel_width = 0.5)
      
      if(is.null(explanation)){
        return(NULL)
      }
      
      valueBox(
        value = randomVals(),
        subtitle = "ID Employee",
        icon = icon("person"),
        color = "orange"
      )
      
    })
    
    output$Label_test <- renderValueBox({
      
      data_test_att <- reactive_att()
      rownames(data_test_att) <- data_test_att$ID_Employee
      em <- data_test_att %>% 
        mutate_if(is.character, factor, ordered = FALSE) %>%
        filter(ID_Employee %in% randomVals()) %>%
        select(Age,BusinessTravel,DistanceFromHome,Education,EducationField,
               EnvironmentSatisfaction,Gender,JobInvolvement,JobLevel,
               JobRole,JobSatisfaction,MaritalStatus,NumCompaniesWorked,
               OverTime,RelationshipSatisfaction,TotalWorkingYears,
               TrainingTimesLastYear,WorkLifeBalance,YearsSinceLastPromotion,
               YearsWithCurrManager) #%>%
      # as_tibble()
      explainer <- lime::lime(
        as.data.frame(train_h2o[,c("Age","BusinessTravel","DistanceFromHome","Education","EducationField",
                                   "EnvironmentSatisfaction","Gender","JobInvolvement","JobLevel",
                                   "JobRole","JobSatisfaction","MaritalStatus","NumCompaniesWorked",
                                   "OverTime","RelationshipSatisfaction","TotalWorkingYears",
                                   "TrainingTimesLastYear","WorkLifeBalance","YearsSinceLastPromotion",
                                   "YearsWithCurrManager")]), 
        model          = dl1, 
        bin_continuous = FALSE)
      
      # Run explain() on explainer
      explanation <- lime::explain(em, 
                                   explainer    = explainer, n_labels = 1,
                                   n_features   = 10,
                                   kernel_width = 0.5)
      
      if(is.null(explanation)){
        return(NULL)
      }
      
      valueBox(
        value = unique(explanation$label),
        subtitle = "Prediction",
        icon = if (unique(explanation$label) == "Yes") icon("check") else icon("xmark"),
        color = if (unique(explanation$label) == "Yes") "red" else "green"
      )
      
    })
    
    output$Probability_test <- renderValueBox({
      
      data_test_att <- reactive_att()
      rownames(data_test_att) <- data_test_att$ID_Employee
      em <- data_test_att %>% 
        mutate_if(is.character, factor, ordered = FALSE) %>%
        filter(ID_Employee %in% randomVals()) %>%
        select(Age,BusinessTravel,DistanceFromHome,Education,EducationField,
               EnvironmentSatisfaction,Gender,JobInvolvement,JobLevel,
               JobRole,JobSatisfaction,MaritalStatus,NumCompaniesWorked,
               OverTime,RelationshipSatisfaction,TotalWorkingYears,
               TrainingTimesLastYear,WorkLifeBalance,YearsSinceLastPromotion,
               YearsWithCurrManager) #%>%
      # as_tibble()
      explainer <- lime::lime(
        as.data.frame(train_h2o[,c("Age","BusinessTravel","DistanceFromHome","Education","EducationField",
                                   "EnvironmentSatisfaction","Gender","JobInvolvement","JobLevel",
                                   "JobRole","JobSatisfaction","MaritalStatus","NumCompaniesWorked",
                                   "OverTime","RelationshipSatisfaction","TotalWorkingYears",
                                   "TrainingTimesLastYear","WorkLifeBalance","YearsSinceLastPromotion",
                                   "YearsWithCurrManager")]), 
        model          = dl1, 
        bin_continuous = FALSE)
      
      # Run explain() on explainer
      explanation <- lime::explain(em, 
                                   explainer    = explainer, n_labels = 1,
                                   n_features   = 10,
                                   kernel_width = 0.5)
      
      if(is.null(explanation)){
        return(NULL)
      }
      
      valueBox(
        value = if_else(unique(explanation$label) == "Yes", 
                        paste0(unique(round(explanation$label_prob,4))*100,"%"), 
                        paste0(100-(unique(round(explanation$label_prob,4))*100),"%")),
        subtitle = "Employee Attrition Probability",
        icon = icon("star"),
        color = "fuchsia"
      )
      
    })
    
    output$plot_att <- renderPlotly({
      
      data_test_att <- reactive_att()
      rownames(data_test_att) <- data_test_att$ID_Employee
      em <- data_test_att %>% 
        mutate_if(is.character, factor, ordered = FALSE) %>%
        filter(ID_Employee %in% randomVals()) %>%
        select(Age,BusinessTravel,DistanceFromHome,Education,EducationField,
               EnvironmentSatisfaction,Gender,JobInvolvement,JobLevel,
               JobRole,JobSatisfaction,MaritalStatus,NumCompaniesWorked,
               OverTime,RelationshipSatisfaction,TotalWorkingYears,
               TrainingTimesLastYear,WorkLifeBalance,YearsSinceLastPromotion,
               YearsWithCurrManager) 
      explainer <- lime::lime(
        as.data.frame(train_h2o[,c("Age","BusinessTravel","DistanceFromHome","Education","EducationField",
                                   "EnvironmentSatisfaction","Gender","JobInvolvement","JobLevel",
                                   "JobRole","JobSatisfaction","MaritalStatus","NumCompaniesWorked",
                                   "OverTime","RelationshipSatisfaction","TotalWorkingYears",
                                   "TrainingTimesLastYear","WorkLifeBalance","YearsSinceLastPromotion",
                                   "YearsWithCurrManager")]), 
        model          = dl1, 
        bin_continuous = FALSE)
      
      # Run explain() on explainer
      explanation <- lime::explain(em, 
                                   explainer    = explainer, labels = "Yes",
                                   n_features   = 10,
                                   kernel_width = 0.5)
      
      if(is.null(explanation)){
        return(NULL)
      }
      
      explain_dl1 <- data.frame(Feature = explanation$feature_desc, 
                                Weight = explanation$feature_weight,
                                Label = glue('Weight : {round(explanation$feature_weight, 3)}'))
      
      plot_individual <- ggplot(mapping = aes(y = reorder(Feature, Weight), 
                                              x = Weight,  
                                              text = Label), data = explain_dl1) +
        geom_col(aes(fill=ifelse(Weight <0,"Contradict", "Support")))+
        scale_fill_manual(
          values = c("Support"="#ce4257","Contradict"="#0077b6"))+
        labs(x = "Weight", y = NaN, 
             title = "Employee Attrition Prediction",
             fill = "Status")+
        theme(legend.position = "left")
      
      plotly_individual <- ggplotly(plot_individual, tooltip = "text") %>% 
        layout(title = list(text = paste0('Employee Attrition Prediction')),
               margin = list(l = 0, r = 0,b = 5, t = 30,pad = 0),
               legend = list(orientation = "h"#, x = -0.5, 
                             ,y =-0.2
               ))
      
      plotly_individual
      
      
      
    })
    
    
}

# Run the application 
shinyApp(ui = ui, server = server)
