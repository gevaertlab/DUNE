suppressMessages({
    library(tidyverse)
    library(data.table)
})

set.seed(123)

var_path <- "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UCSF/metadata/0-variable_list.csv"
raw_meta_path <- "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UCSF/metadata/UCSF_metadata.csv"
list_patients <- list.files("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UCSF/images")



varlist <- fread(var_path)
catg_cols <- varlist$var[varlist$task == "classification"]
reg_cols <- varlist$var[varlist$task == "regression"]
surv_cols <- c("death_delay", "death_event")



raw <- fread(raw_meta_path) %>%
    filter(ID %in% list_patients)

export <- raw %>%
    column_to_rownames("ID") %>%
    select(-c("MGMT_status", "MGMT_index", "1p_19q", "EOR", "BraTS21_ID", "BraTS21_Segmentation_Cohort", "BraTS21_MGMT_Cohort"))

export$Sex <- ordered(export$Sex, levels = c("F", "M"))
export$grade <- ordered(export$grade, levels=c(2,3,4))
export$final_pathology_diag <- ordered(export$final_pathology_diag, levels = c("Astrocytoma, IDH-wildtype", "Astrocytoma, IDH-mutant", "Glioblastoma, IDH-wildtype", "Oligodendroglioma, IDH-mutant, 1p/19q-codeleted"))
# export$MGMT_status <- ordered(export$MGMT_status, levels=c("negative","positive"))
export$IDH <- factor(export$IDH)
export$IDH <- relevel(export$IDH, "wildtype")
export$IDH <- ordered(export$IDH)
export$IDH_bin <- ordered(export$IDH %in% c("wildtype", "mutated (NOS)"), levels = c("FALSE", "TRUE"))
export$Biopsy_prior_to_imaging <- ordered(export$Biopsy_prior_to_imaging, levels = c("No", "Yes"))

export <- export  %>% filter(!is.na(death_delay))

# ENCODING





encoded <- export %>% mutate_if(is.ordered, as.integer)
encoded[, catg_cols] <- encoded[, catg_cols] - 1

# NORMALISATION

normalized <- encoded
normalized[, reg_cols] <- scale(normalized[, reg_cols])
normalized <- normalized %>% rownames_to_column("eid")



normalized$cohort <- sample(c("train", "test"), nrow(normalized), replace = TRUE, prob = c(0.8, 0.2))


fwrite(normalized,
    file = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UCSF/metadata/0-UCSF_metadata_encoded.csv"
)
