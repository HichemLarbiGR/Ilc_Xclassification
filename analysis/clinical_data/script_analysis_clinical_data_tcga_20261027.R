# =============================================================================
# TCGA-BRCA: clinical data preparation for ILC vs. NST classification
# -----------------------------------------------------------------------------
# Purpose : Clean TCGA-BRCA clinical tables, define the histological subtype
#           target (ILC vs. NST), check available UNI2 slide embeddings, build
#           the train/test split and export descriptive tables and JSON files
#           for the model dataloader.
# Inputs  : data/*.tsv (GDC clinical exports), UNI2 embedding id/file lists
# Outputs : output/*.html (summary tables), output/*.json (split and targets)
# =============================================================================


# 0. SETUP ----------------------------------------------------------------

library(tidyverse)
library(gtsummary)
library(labelled)
library(gt)
library(jsonlite)

theme_gtsummary_journal(journal = "jama")
theme_gtsummary_compact()
theme_gtsummary_language("en")

set.seed(42)


# 1. CONSTANTS ------------------------------------------------------------

# Target encoding: ILC = 1, NST = 0
ILC_CLASSES     <- c("ILC", "NST")
ILC_CLASS_CODES <- c(1, 0)

# Variable labels used in the summary tables
var_labels <- list(
  # Demographics
  demographic.age_at_index               = "Age at index (years)",
  demographic.age_is_obfuscated          = "Age obfuscated",
  demographic.race                       = "Race",
  demographic.sex_at_birth               = "Sex at birth",
  demographic.vital_status               = "Vital status",
  
  # Diagnoses
  diagnoses.age_at_diagnosis             = "Age at diagnosis (years)",
  diagnoses.ajcc_pathologic_m            = "AJCC pathologic M",
  diagnoses.ajcc_pathologic_n            = "AJCC pathologic N",
  diagnoses.ajcc_pathologic_stage        = "AJCC pathologic stage",
  diagnoses.ajcc_pathologic_t            = "AJCC pathologic T",
  diagnoses.ajcc_staging_system_edition  = "AJCC staging system edition",
  diagnoses.classification_of_tumor      = "Classification of tumor",
  diagnoses.diagnosis_is_primary_disease = "Diagnosis is primary disease",
  diagnoses.laterality                   = "Laterality",
  diagnoses.metastasis_at_diagnosis      = "Metastasis at diagnosis",
  diagnoses.method_of_diagnosis          = "Method of diagnosis",
  diagnoses.primary_diagnosis            = "Primary diagnosis",
  diagnoses.prior_malignancy             = "Prior malignancy",
  diagnoses.prior_treatment              = "Prior treatment",
  diagnoses.progression_or_recurrence    = "Progression or recurrence",
  diagnoses.site_of_resection_or_biopsy  = "Site of resection or biopsy",
  diagnoses.sites_of_involvement         = "Sites of involvement",
  diagnoses.synchronous_malignancy       = "Synchronous malignancy",
  diagnoses.tissue_or_organ_of_origin    = "Tissue or organ of origin",
  diagnoses.tumor_grade                  = "Tumor grade",
  diagnoses.tumor_of_origin              = "Tumor of origin",
  diagnoses.year_of_diagnosis            = "Year of diagnosis",
  
  # Treatments
  treatments.clinical_trial_indicator    = "Clinical trial indicator",
  treatments.course_number               = "Course number",
  treatments.days_to_treatment_end       = "Days to treatment end",
  treatments.days_to_treatment_start     = "Days to treatment start",
  treatments.initial_disease_status      = "Initial disease status",
  treatments.margin_status               = "Margin status",
  treatments.number_of_cycles            = "Number of cycles",
  treatments.number_of_fractions         = "Number of fractions",
  treatments.prescribed_dose             = "Prescribed dose",
  treatments.regimen_or_line_of_therapy  = "Regimen or line of therapy",
  treatments.route_of_administration     = "Route of administration",
  treatments.therapeutic_agents          = "Therapeutic agents",
  treatments.treatment_anatomic_sites    = "Treatment anatomic sites",
  treatments.treatment_dose              = "Treatment dose",
  treatments.treatment_intent_type       = "Treatment intent",
  treatments.treatment_or_therapy        = "Treatment or therapy",
  treatments.treatment_outcome           = "Treatment outcome",
  treatments.treatment_type              = "Treatment type",
  
  # Group / project
  ilc.classes                            = "Histological subtype",
  project.project_id                     = "Project ID"
)


# 2. HELPER FUNCTIONS -----------------------------------------------------

# Descriptive table with continuous summaries (median [IQR], mean [SD], range)
# and categorical counts (n [%]), stratified by `by_var`.
build_summary_table <- function(data, by_var) {
  data %>%
    tbl_summary(
      by = {{ by_var }},
      type = all_continuous() ~ "continuous2",
      statistic = list(
        all_continuous2() ~ c("{median} ({p25} - {p75})", "{mean} ({sd})", "{min} - {max}"),
        all_categorical() ~ "{n} ({p})"
      ),
      digits = all_continuous2() ~ 1
    ) %>%
    bold_labels()
}


# 3. DATA IMPORT ----------------------------------------------------------

clinical_data       <- read.csv2("data/clinical.tsv",         sep = "\t", na.strings = "'--")
exposure_data       <- read.csv2("data/exposure.tsv",         sep = "\t", na.strings = "'--")
family_history_data <- read.csv2("data/family_history.tsv",   sep = "\t", na.strings = "'--")  # Not informative
followup_data       <- read.csv2("data/follow_up.tsv",        sep = "\t", na.strings = "'--")
pathology_data      <- read.csv2("data/pathology_detail.tsv", sep = "\t", na.strings = "'--")

uni2_ids_data       <- read.csv2("data/ids_emb_uni2.txt",       header = FALSE)
uni2_emb_wsi_files  <- read.csv2("data/list_wsi_UNI2_emb.txt",  header = FALSE)


# 4. EXPLORATORY DATA ANALYSIS --------------------------------------------

## 4.1 Table dimensions and structure
clinical_data       %>% dim()
exposure_data       %>% dim()
family_history_data %>% dim()
followup_data       %>% dim()
pathology_data      %>% dim()

clinical_data       %>% glimpse()
exposure_data       %>% glimpse()
family_history_data %>% glimpse()
followup_data       %>% glimpse()
pathology_data      %>% glimpse()

## 4.2 Missingness overview
naniar::vis_miss(clinical_data %>% select(starts_with("demographic")))
naniar::vis_miss(clinical_data %>% select(starts_with("diagnoses")))
naniar::vis_miss(clinical_data %>% select(starts_with("treatments")))

## 4.3 Columns that are entirely missing, per table
get_empty_columns <- function(data) {
  data %>%
    naniar::miss_var_summary() %>%
    filter(pct_miss == 100) %>%
    select(variable) %>%
    pull()
}

empty_cols_clinical       <- get_empty_columns(clinical_data)
empty_cols_exposure       <- get_empty_columns(exposure_data)
empty_cols_family_history <- get_empty_columns(family_history_data)
empty_cols_followup       <- get_empty_columns(followup_data)
empty_cols_pathology      <- get_empty_columns(pathology_data)

## 4.4 Quick overview of the raw clinical table
clinical_data %>%
  select(-starts_with("cases"), -contains("_id"), project.project_id) %>%
  tbl_summary()


# 5. DATA CLEANING --------------------------------------------------------

## 5.1 Clinical table: drop empty / unused columns, derive the ILC/NST target
clinical_data_cleaned <- clinical_data %>%
  
  set_variable_labels(!!!var_labels[intersect(names(var_labels), names(.))]) %>%
  
  select(
    -empty_cols_clinical,
    -demographic.country_of_residence_at_enrollment,
    -demographic.days_to_birth,
    -demographic.days_to_death,
    -demographic.ethnicity,
    -demographic.year_of_birth,
    -diagnoses.days_to_diagnosis,
    -diagnoses.days_to_last_follow_up,
    -diagnoses.figo_stage,
    -diagnoses.figo_staging_edition_year,
    -diagnoses.icd_10_code,
    -diagnoses.last_known_disease_status,
    -treatments.prescribed_dose_units,
    -treatments.treatment_dose_units,
    -treatments.treatment_type,
    -treatments.therapeutic_agents,
    -diagnoses.tumor_of_origin
  ) %>%
  
  mutate(
    # ICD-O-3 morphology codes: 8500/3 = NST, 8520/3 = ILC
    ilc.classes = case_when(
      diagnoses.morphology == "8500/3" ~ "NST",
      diagnoses.morphology == "8520/3" ~ "ILC",
      T ~ as.character(NA)
    ),
    # Convert age at diagnosis from days to years
    diagnoses.age_at_diagnosis     = diagnoses.age_at_diagnosis / 365,
    
    treatments.number_of_fractions = as.numeric(treatments.number_of_fractions),
    treatments.prescribed_dose     = as.numeric(treatments.prescribed_dose)
  ) %>%
  
  mutate(
    ilc.classes     = factor(ilc.classes, levels = ILC_CLASSES),
    ilc.classes.ord = ILC_CLASS_CODES[match(as.character(ilc.classes), ILC_CLASSES)]
  ) %>%
  
  filter(ilc.classes %in% c("ILC", "NST")) %>%
  select(-diagnoses.morphology)

## 5.2 Other tables
exposure_data        # Not processed: almost empty
family_history_data  # Not processed: almost empty

followup_data_cleaned  <- followup_data  %>% select(-empty_cols_followup)   # Not processed further
pathology_data_cleaned <- pathology_data %>% select(-empty_cols_pathology)  # Apart from MSI, no relevant variables

uni2_ids_data_cleaned <- uni2_ids_data %>% rename(cases.submitter_id = "V1")


# 6. PATIENT-LEVEL DIAGNOSIS CHECK ----------------------------------------

# Identify patients assigned to more than one histological subtype
ambiguity_check <- clinical_data_cleaned %>%
  group_by(cases.submitter_id) %>%
  summarise(n_classes = n_distinct(ilc.classes), .groups = "drop") %>%
  filter(n_classes > 1)

ambiguous_cases <- ambiguity_check$cases.submitter_id

nrow(ambiguity_check)

clinical_data_cleaned %>%
  filter(cases.submitter_id %in% ambiguous_cases) %>%
  View()

clinical_data_cleaned %>%
  filter(cases.submitter_id %in% ambiguous_cases) %>%
  group_by(cases.submitter_id) %>%
  summarise(
    n_rows          = n(),
    n_diagnosis_id  = n_distinct(diagnoses.diagnosis_id),
    n_primary_true  = sum(diagnoses.diagnosis_is_primary_disease == "true", na.rm = TRUE),
    n_classes       = n_distinct(ilc.classes),
    classes         = paste(unique(ilc.classes), collapse = " / "),
    classifications = paste(unique(diagnoses.classification_of_tumor), collapse = " / "),
    .groups = "drop"
  ) %>%
  arrange(desc(n_classes))

# CONCLUSION: exclude the 5 ambiguous cases
clinical_data_cleaned_no_ambiguity <- clinical_data_cleaned %>%
  filter(!cases.submitter_id %in% ambiguous_cases)

clinical_data_cleaned_no_ambiguity %>% dim()


# 7. CONSISTENCY WITH EXTRACTED EMBEDDINGS --------------------------------

# Patients with embeddings but absent from the cleaned clinical data
uni2_ids_data_cleaned$cases.submitter_id[
  !uni2_ids_data_cleaned$cases.submitter_id %in% clinical_data_cleaned_no_ambiguity$cases.submitter_id
]

# Patients in the cleaned clinical data without embeddings
clinical_data_cleaned_no_ambiguity$cases.submitter_id[
  !clinical_data_cleaned_no_ambiguity$cases.submitter_id %in% uni2_ids_data_cleaned$cases.submitter_id
]

# Keep patients present in both sources
ids_model <- uni2_ids_data_cleaned$cases.submitter_id[
  uni2_ids_data_cleaned$cases.submitter_id %in% clinical_data_cleaned_no_ambiguity$cases.submitter_id
]

# Map each embedding file to its patient (TCGA barcode)
wsi_emb_data <- uni2_emb_wsi_files %>%
  rename(filename = "V1") %>%
  mutate(cases.submitter_id = str_extract(filename, "TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}"))


# 8. BIVARIATE ANALYSIS (ILC vs. NST) -------------------------------------

table_clinical <- clinical_data_cleaned %>%
  mutate(
    treatments.clinical_trial_indicator   = factor(treatments.clinical_trial_indicator),
    treatments.course_number              = factor(treatments.course_number),
    treatments.regimen_or_line_of_therapy = factor(treatments.regimen_or_line_of_therapy),
    treatments.route_of_administration    = factor(treatments.route_of_administration),
    treatments.treatment_outcome          = factor(treatments.treatment_outcome)
  ) %>%
  filter(treatments.initial_disease_status == "Initial Diagnosis") %>%
  select(-starts_with("cases"), -contains("_id"), -project.project_id,
         -diagnoses.progression_or_recurrence,
         -diagnoses.tumor_grade) %>%
  tbl_summary(
    by = ilc.classes,
    type = all_continuous() ~ "continuous2",
    statistic = list(
      all_continuous2() ~ c("{median} ({p25} - {p75})", "{mean} ({sd})", "{min} - {max}"),
      all_categorical() ~ "{n} ({p})"
    ),
    digits = all_continuous2() ~ 1
  ) %>%
  bold_labels()

table_clinical


# 9. TRAIN / TEST SPLIT ---------------------------------------------------

# Random 80/20 split at the patient level
split_train_test <- sample(c(TRUE, FALSE), length(ids_model), replace = TRUE, prob = c(0.8, 0.2))

ids_model_train <- ids_model[split_train_test == TRUE]
ids_model_test  <- ids_model[split_train_test == FALSE]

ids_model_train %>% length()
ids_model_test  %>% length()

clinical_data_split <- clinical_data_cleaned %>%
  filter(treatments.initial_disease_status == "Initial Diagnosis") %>%
  filter(cases.submitter_id %in% ids_model) %>%
  ungroup() %>%
  mutate(
    Split = case_when(
      cases.submitter_id %in% ids_model_train ~ "Train",
      cases.submitter_id %in% ids_model_test  ~ "Test"
    )
  )

table_clinical_split <- clinical_data_split %>%
  filter(treatments.initial_disease_status == "Initial Diagnosis") %>%
  select(where(~ !all(is.na(.x)))) %>%
  select(-starts_with("cases"), -contains("_id"), -project.project_id) %>%
  tbl_summary(
    by = Split,
    type = all_continuous() ~ "continuous2",
    statistic = list(
      all_continuous2() ~ c("{median} ({p25} - {p75})", "{mean} ({sd})", "{min} - {max}"),
      all_categorical() ~ "{n} ({p})"
    ),
    digits = all_continuous2() ~ 1
  ) %>%
  bold_labels()

table_clinical_split


# 10. DATALOADER FILES ----------------------------------------------------

json_list_train_test <- list(
  train_val = clinical_data_split %>% filter(Split == "Train") %>% select(cases.submitter_id) %>% distinct %>% pull(),
  test      = clinical_data_split %>% filter(Split == "Test")  %>% select(cases.submitter_id) %>% distinct %>% pull()
)

json_targets <- as.list(
  clinical_data_split %>%
    select(cases.submitter_id, ilc.classes.ord) %>%
    distinct %>%
    tibble::deframe()
)


# 11. EXPORT --------------------------------------------------------------

table_clinical %>%
  as_gt() %>%
  tab_source_note(source_note = "Only the initial diagnosis was considered") %>%
  gtsave(filename = "output/table_clinical_data_20261028.html")

table_clinical_split %>%
  as_gt() %>%
  tab_source_note(source_note = "Only the initial diagnosis was considered") %>%
  gtsave(filename = "output/table_clinical_split_20261028.html")

write_json(json_list_train_test, "output/split_ids.json",     pretty = TRUE, auto_unbox = TRUE)
write_json(json_targets,         "output/split_targets.json", pretty = TRUE, auto_unbox = TRUE)