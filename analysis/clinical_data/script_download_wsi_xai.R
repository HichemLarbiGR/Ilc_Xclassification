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

set.seed(42)


# 1. CONSTANTS ------------------------------------------------------------

wsi_good_pred  = c("TCGA-A2-A04Y","TCGA-A7-A426","TCGA-AC-A2FO","TCGA-AC-A8OS","TCGA-E9-A227")

wsi_wrong_pred = c("TCGA-C8-A3M8","TCGA-A2-A0EW","TCGA-GM-A2DO","TCGA-AR-A1AO","TCGA-HN-A2NL")


# 3. DATA IMPORT ----------------------------------------------------------

manifest_wsi = read.csv2("data/slide_manifest.csv", sep=",")
json_test    = read_json("output/dataset_asier.json")


# 4. EXPLORATORY DATA ANALYSIS --------------------------------------------

## 4.1 Table dimensions and structure
manifest_wsi       %>% dim()
manifest_wsi       %>% glimpse()


## 4.2 Missingness overview
naniar::vis_miss(manifest_wsi)

## 4.3 Columns that are entirely missing, per table
get_empty_columns <- function(data) {
  data %>%
    naniar::miss_var_summary() %>%
    filter(pct_miss == 100) %>%
    select(variable) %>%
    pull()
}

empty_cols_manifest       <- get_empty_columns(manifest_wsi)

## 4.4 Quick overview of the raw clinical table
manifest_wsi %>%
  select(-starts_with("cases"), -contains("_id")) %>%
  tbl_summary()


# 5. DATA CLEANING --------------------------------------------------------

## 5.1 Clinical table: drop empty / unused columns, derive the ILC/NST target
manifest_wsi_pred = manifest_wsi %>% filter(patient_id %in% wsi_good_pred | patient_id %in% wsi_wrong_pred) %>%
  mutate(correct_pred = ifelse(patient_id %in% wsi_good_pred, TRUE, FALSE))

## 5.2 Other tables
gdc_ids_to_download = manifest_wsi_pred$gdc_file_id

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
  mutate(cases.submitter_id = str_extract(filename, "TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}"),
         piv_col = ifelse(grepl("DX1", filename),"DX1", "DX2"),
         filename = gsub("UNI2-h_features/TCGA/", "", filename)) %>%
  pivot_wider(names_from = piv_col, values_from = filename) %>%
  mutate(
    DX1 = map_chr(DX1, ~ if (is.null(.x) || length(.x) == 0) NA_character_ else paste(.x, collapse = ";")),
    DX2 = map_chr(DX2, ~ if (is.null(.x) || length(.x) == 0) NA_character_ else paste(.x, collapse = ";"))
  ) %>%
  
  mutate(DX1 = ifelse(is.na(DX1),DX2,DX1))

wsi_emb_data %>% naniar::miss_var_summary()

#Comment : 94% of DX2 are missing

# 7bis. CHECK PROF. ASIER EMBEDDINGS FILES --------------------------------


wsi_hopt0_emb_data <- asier_emb_wsi_files %>%
  rename(filename = "V1") %>%
  mutate(cases.submitter_id = str_extract(filename, "TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}"),
         piv_col = ifelse(grepl("DX1", filename),"DX1", "DX2"),
         filename = gsub("UNI2-h_features/TCGA/", "", filename)) %>%
  pivot_wider(names_from = piv_col, values_from = filename) %>%
  mutate(
    DX1 = map_chr(DX1, ~ if (is.null(.x) || length(.x) == 0) NA_character_ else paste(.x, collapse = ";")),
    DX2 = map_chr(DX2, ~ if (is.null(.x) || length(.x) == 0) NA_character_ else paste(.x, collapse = ";"))
  ) %>%
  
  mutate(DX1 = ifelse(is.na(DX1),DX2,DX1))

wsi_hopt0_emb_data %>% naniar::miss_var_summary()


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
  build_summary_table(ilc.classes)

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
  build_summary_table(Split)

table_clinical_split

# 9. bis Download wsi ----------------------------------------------------

wsi_good_pred  = c()

wsi_wrong_pred = c()

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

json_wsi_files <- as.list(
  wsi_emb_data %>% filter(cases.submitter_id %in% ids_model_train | cases.submitter_id %in% ids_model_test) %>%
    select(cases.submitter_id,DX1) %>%
    tibble::deframe()
)

json_dataset <- list(
  train_val = clinical_data_split %>%
    filter(Split == "Train") %>%
    select(cases.submitter_id, ilc.classes.ord) %>%
    distinct %>%
    left_join(
      wsi_emb_data %>% select(cases.submitter_id, DX1),
      by = "cases.submitter_id"
    ) %>%
    filter(!is.na(DX1)) %>%
    transmute(id = cases.submitter_id, filename = DX1, target = ilc.classes.ord) %>%
    purrr::transpose(),
  
  test = clinical_data_split %>%
    filter(Split == "Test") %>%
    select(cases.submitter_id, ilc.classes.ord) %>%
    distinct %>%
    left_join(
      wsi_emb_data %>% select(cases.submitter_id, DX1),
      by = "cases.submitter_id"
    ) %>%
    filter(!is.na(DX1)) %>%
    transmute(id = cases.submitter_id, filename = DX1, target = ilc.classes.ord) %>%
    purrr::transpose()
)


json_dataset_asier <- list(
  train_val = clinical_data_split %>%
    filter(Split == "Train") %>%
    select(cases.submitter_id, ilc.classes.ord) %>%
    distinct %>%
    left_join(
      wsi_hopt0_emb_data %>% select(cases.submitter_id, DX1),
      by = "cases.submitter_id"
    ) %>%
    filter(!is.na(DX1)) %>%
    transmute(id = cases.submitter_id, filename = DX1, target = ilc.classes.ord) %>%
    purrr::transpose(),
  
  test = clinical_data_split %>%
    filter(Split == "Test") %>%
    select(cases.submitter_id, ilc.classes.ord) %>%
    distinct %>%
    left_join(
      wsi_hopt0_emb_data %>% select(cases.submitter_id, DX1),
      by = "cases.submitter_id"
    ) %>%
    filter(!is.na(DX1)) %>%
    transmute(id = cases.submitter_id, filename = DX1, target = ilc.classes.ord) %>%
    purrr::transpose()
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


write_json(json_list_train_test, "output/split_ids.json",      pretty = TRUE, auto_unbox = TRUE)
write_json(json_targets,         "output/split_targets.json",  pretty = TRUE, auto_unbox = TRUE)
write_json(json_wsi_files,       "output/list_wsi_train.json", pretty = TRUE, auto_unbox = TRUE)
write_json(json_dataset,         "output/dataset.json",        pretty = TRUE, auto_unbox = TRUE)
write_json(json_dataset_asier,   "output/dataset_asier.json",  pretty = TRUE, auto_unbox = TRUE)
