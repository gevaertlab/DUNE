suppressMessages({
    library(tidyverse)
    library(data.table)
})



varlist <- fread("/home/tbarba/projects/MultiModalBrainSurvival/data/metadata/UCSF/0-variable_list.csv")

catg_cols <- varlist$var[varlist$task == "classification"]
reg_cols <- varlist$var[varlist$task == "regression"]
surv_cols <- c("death_delay", "death_event")




raw <- fread("/home/tbarba/projects/MultiModalBrainSurvival/data/metadata/UCSF/UCSF_metadata.csv")

export <- raw %>%
    column_to_rownames("ID")

export$Sex <- ordered(export$Sex, levels = c("F", "M"))
export$grade <- ordered(export$grade, levels=c(2,3,4))
export$final_pathology_diag <- ordered(export$final_pathology_diag, levels = c("Astrocytoma, IDH-wildtype", "Astrocytoma, IDH-mutant", "Glioblastoma, IDH-wildtype", "Oligodendroglioma, IDH-mutant, 1p/19q-codeleted"))
# export$MGMT_status <- ordered(export$MGMT_status, levels=c("negative","positive"))
export$IDH <- factor(export$IDH)
export$IDH <- relevel(export$IDH, "wildtype")
export$IDH <- ordered(export$IDH)
export$IDH_bin <- ordered(export$IDH=="wildtype", levels=c("FALSE","TRUE"))
export$Biopsy_prior_to_imaging <- ordered(export$Biopsy_prior_to_imaging, levels = c("No", "Yes"))

export <- export  %>% filter(!is.na(death_delay))

# ENCODING





encoded <- export %>% mutate_if(is.ordered, as.integer)
encoded[, catg_cols] <- encoded[, catg_cols] - 1

# NORMALISATION

normalized <- encoded
normalized[, reg_cols] <- scale(normalized[, reg_cols])
normalized <- normalized %>% rownames_to_column("eid")


fwrite(normalized,
    file = "/home/tbarba/projects/MultiModalBrainSurvival/data/metadata/UCSF/0-UCSF_metadata_encoded.csv"
)
