# LIBRARIES ----
library(tidyverse)
library(readxl)
library(gtsummary)
library(plotly)
library(labelled)
library(performance)
library(survival)
library(survminer)  # Pour les superbes graphiques de KM
library(flexsurv)   # Pour le modèle Weibull (plus robuste que le Cox pour les pics à 0)
library(tidyverse)
library(ggiraph)

####################################

# TCGA BRCA clinical data

###################################

theme_gtsummary_journal(journal = "jama")
theme_gtsummary_compact()
theme_gtsummary_language("fr")


# DATA IMPORT ----

clinical_data = read.csv2("data/clinical.tsv", sep = "\t", na.strings = "'--")
exposure_data = read.csv2("data/exposure.tsv", sep = "\t", na.strings = "'--")
family_history_data = read.csv2("data/family_history.tsv", sep = "\t", na.strings = "'--") #useless
followup_data = read.csv2("data/follow_up.tsv", sep = "\t", na.strings = "'--")
pathology_data = read.csv2("data/pathology_detail.tsv", sep = "\t", na.strings = "'--")
uni2_ids_data = read.csv2("data/ids_emb_uni2.txt", header = FALSE)

# FUNCTIONS & CONSTANT ----

ILS_class = c("ILC", "NST")
ILS_ord = c(1, 0)

library(dplyr)
library(labelled)
library(gtsummary)

# Labels for ALL variables in the table
var_labels <- list(
  # Demographics
  demographic.age_at_index              = "Age at index (years)",
  demographic.age_is_obfuscated         = "Age obfuscated",
  demographic.race                      = "Race",
  demographic.sex_at_birth              = "Sex at birth",
  demographic.vital_status              = "Vital status",
  
  # Diagnoses
  diagnoses.age_at_diagnosis            = "Age at diagnosis (years)",
  diagnoses.ajcc_pathologic_m           = "AJCC pathologic M",
  diagnoses.ajcc_pathologic_n           = "AJCC pathologic N",
  diagnoses.ajcc_pathologic_stage       = "AJCC pathologic stage",
  diagnoses.ajcc_pathologic_t           = "AJCC pathologic T",
  diagnoses.ajcc_staging_system_edition = "AJCC staging system edition",
  diagnoses.classification_of_tumor     = "Classification of tumor",
  diagnoses.diagnosis_is_primary_disease= "Diagnosis is primary disease",
  diagnoses.laterality                  = "Laterality",
  diagnoses.metastasis_at_diagnosis     = "Metastasis at diagnosis",
  diagnoses.method_of_diagnosis         = "Method of diagnosis",
  diagnoses.primary_diagnosis           = "Primary diagnosis",
  diagnoses.prior_malignancy            = "Prior malignancy",
  diagnoses.prior_treatment             = "Prior treatment",
  diagnoses.progression_or_recurrence   = "Progression or recurrence",
  diagnoses.site_of_resection_or_biopsy = "Site of resection or biopsy",
  diagnoses.sites_of_involvement        = "Sites of involvement",
  diagnoses.synchronous_malignancy      = "Synchronous malignancy",
  diagnoses.tissue_or_organ_of_origin   = "Tissue or organ of origin",
  diagnoses.tumor_grade                 = "Tumor grade",
  diagnoses.tumor_of_origin             = "Tumor of origin",
  diagnoses.year_of_diagnosis           = "Year of diagnosis",
  
  # Treatments
  treatments.clinical_trial_indicator   = "Clinical trial indicator",
  treatments.course_number              = "Course number",
  treatments.days_to_treatment_end      = "Days to treatment end",
  treatments.days_to_treatment_start    = "Days to treatment start",
  treatments.initial_disease_status     = "Initial disease status",
  treatments.margin_status              = "Margin status",
  treatments.number_of_cycles           = "Number of cycles",
  treatments.number_of_fractions        = "Number of fractions",
  treatments.prescribed_dose            = "Prescribed dose",
  treatments.regimen_or_line_of_therapy = "Regimen or line of therapy",
  treatments.route_of_administration    = "Route of administration",
  treatments.therapeutic_agents         = "Therapeutic agents",
  treatments.treatment_anatomic_sites   = "Treatment anatomic sites",
  treatments.treatment_dose             = "Treatment dose",
  treatments.treatment_intent_type      = "Treatment intent",
  treatments.treatment_or_therapy       = "Treatment or therapy",
  treatments.treatment_outcome          = "Treatment outcome",
  treatments.treatment_type             = "Treatment type",
  
  # Group / project
  ilc.classes                           = "Histological subtype",
  project.project_id                    = "Project ID"
)

# EDA ----

clinical_data %>% dim()
exposure_data %>% dim()
family_history_data %>% dim()
followup_data %>% dim()
pathology_data %>% dim()

clinical_data %>% glimpse() 
exposure_data %>% glimpse()
family_history_data %>% glimpse()
followup_data %>% glimpse()
pathology_data %>% glimpse()

naniar::vis_miss(clinical_data %>% select(starts_with("demographic")))
naniar::vis_miss(clinical_data %>% select(starts_with("diagnoses")))
naniar::vis_miss(clinical_data %>% select(starts_with("treatments")))
naniar::vis_miss(exposure_data)
naniar::vis_miss(family_history_data)
naniar::vis_miss(followup_data)
naniar::vis_miss(pathology_data)


#Missing variables 


var_miss_clinical_data = clinical_data %>% naniar::miss_var_summary() %>% filter(pct_miss==100) %>% select(variable) %>% pull()
var_miss_exposure_data = exposure_data %>% naniar::miss_var_summary() %>% filter(pct_miss==100) %>% select(variable) %>% pull()
var_miss_family_history_data = family_history_data %>% naniar::miss_var_summary() %>% filter(pct_miss==100) %>% select(variable) %>% pull()
var_miss_followup_data = followup_data %>% naniar::miss_var_summary() %>% filter(pct_miss==100) %>% select(variable) %>% pull()
var_miss_pathology_data = pathology_data %>% naniar::miss_var_summary() %>% filter(pct_miss==100) %>% select(variable) %>% pull()

clinical_data %>% select(-starts_with("cases"), -contains("_id"), project.project_id) %>% tbl_summary()


# CLEANING ----

clinical_data_cleaned = clinical_data %>% 
  
  set_variable_labels(!!!var_labels[intersect(names(var_labels), names(.))]) %>%
  
  select(-var_miss_clinical_data, -demographic.country_of_residence_at_enrollment, 
         -demographic.days_to_birth, -demographic.days_to_death, -demographic.ethnicity, 
         -demographic.year_of_birth, -diagnoses.days_to_diagnosis, -diagnoses.days_to_last_follow_up,
         -diagnoses.figo_stage, -diagnoses.figo_staging_edition_year, -diagnoses.icd_10_code,
         -diagnoses.last_known_disease_status, -treatments.prescribed_dose_units,-treatments.treatment_dose_units,
         -treatments.treatment_type, -treatments.therapeutic_agents, -diagnoses.tumor_of_origin) %>% 
  
  mutate(ilc.classes = case_when(diagnoses.morphology=="8500/3" ~ "NST",
                                 diagnoses.morphology=="8520/3" ~ "ILC",
                                 T ~ as.character(NA)),
         diagnoses.age_at_diagnosis = diagnoses.age_at_diagnosis/365,
         
         treatments.number_of_fractions = as.numeric(treatments.number_of_fractions),
         treatments.prescribed_dose = as.numeric(treatments.prescribed_dose)
         ) %>%
  
  mutate(ilc.classes = factor(ilc.classes, levels = ILS_class),
         ilc.classes.ord = ILS_ord[match(as.character(ilc.classes), ILS_class)]) %>% 
  
  filter(ilc.classes %in% c("ILC", "NST")) %>% select(-diagnoses.morphology)

exposure_data  # No need to process this table, almost empty

family_history_data  # No need to process this table, almost empty

followup_data_cleaned = followup_data %>% select(-var_miss_followup_data) # No need to process this table 

pathology_data_cleaned = pathology_data %>% select(-var_miss_pathology_data) #Apart from MSI, no relevant variables

uni2_ids_data_cleaned = uni2_ids_data %>% rename(cases.submitter_id = "V1")

#Check patient diagnostic

check <- clinical_data_cleaned %>%
  group_by(cases.submitter_id) %>%
  summarise(n_classes = n_distinct(ilc.classes), .groups = "drop") %>%
  filter(n_classes > 1)

ambigous_cases = check$cases.submitter_id 

nrow(check)

clinical_data_cleaned %>% filter(cases.submitter_id %in% ambigous_cases) %>% View()

clinical_data_cleaned %>%
  filter(cases.submitter_id %in% ambigous_cases) %>%
  group_by(cases.submitter_id) %>%
  summarise(
    n_rows             = n(),
    n_diagnosis_id     = n_distinct(diagnoses.diagnosis_id),
    n_primary_true     = sum(diagnoses.diagnosis_is_primary_disease == "true", na.rm = TRUE),
    n_classes          = n_distinct(ilc.classes),
    classes            = paste(unique(ilc.classes), collapse = " / "),
    classifications    = paste(unique(diagnoses.classification_of_tumor), collapse = " / "),
    .groups = "drop"
  ) %>%
  arrange(desc(n_classes))

#CONCLUSION : exclude 5 cases --> ambigous_cases

clinical_data_cleaned_from_ambigous = clinical_data_cleaned %>% filter(!cases.submitter_id %in% ambigous_cases)

clinical_data_cleaned_from_ambigous %>% dim()

#CHECK EMBEDDINGS EXTRACTED

uni2_ids_data_cleaned$cases.submitter_id[uni2_ids_data_cleaned$cases.submitter_id %in% clinical_data_cleaned_from_ambigous$cases.submitter_id==FALSE]

clinical_data_cleaned_from_ambigous$cases.submitter_id[clinical_data_cleaned_from_ambigous$cases.submitter_id %in% uni2_ids_data_cleaned$cases.submitter_id ==FALSE]

#We keep those ids 
ids_model = uni2_ids_data_cleaned$cases.submitter_id[uni2_ids_data_cleaned$cases.submitter_id %in% clinical_data_cleaned_from_ambigous$cases.submitter_id==TRUE] %>% length()


# ANALYSIS ----

## UNIVARIATE ANALYSIS----


table_clinical = clinical_data_cleaned %>% select(-starts_with("cases"), -contains("_id"), project.project_id) %>% 
  

tbl_summary(  by = ilc.classes,
              statistic = list(
                all_continuous2()  ~ c("{median} ({p25} - {p75})", "{mean} ({sd})", "{min} - {max}"),
                all_categorical() ~ "{n} ({p})"
                ),
  digits = all_continuous() ~ 1
) %>%
  bold_labels()


## TRAINING ----

set.seed(42)

clinical_data %>% select(cases.case_id) %>% dim()

clinical_data_cleaned %>% select(contains("id")) %>% View()

clinical_data_cleaned %>% select(cases.case_id, ilc.classes) %>% View()

pathology_data_cleaned %>% dim()


#EXPORT ----

table_clinical %>%
  as_gt() %>%
  gtsave(filename = "table_clinical_data_20261027.html")
