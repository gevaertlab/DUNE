# NB. Fichier principal de formattage et de prétraitement de la base se situe ici : /home/tbarba/projects/MultiModalBrainSurvival/data/ukbiobank2020/update/data/run41271/metadata/merge/merge_ukb41271.r


suppressMessages({
    library(tidyverse)
    library(data.table)
})

set.seed(123)

var_path <- "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UKBIOBANK/metadata/0-variable_list.csv"
raw_meta_path <- "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UKBIOBANK/metadata/0-imputed.csv.gz"
list_patients <- list.files("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UKBIOBANK/images")



varlist <- fread(var_path)
catg_cols <- varlist$var[varlist$task == "classification"]
reg_cols <- varlist$var[varlist$task == "regression"]
surv_cols <- c("death_delay", "death_event")



raw <- fread(raw_meta_path) %>%
    filter(eid %in% list_patients)
raw$APOE_homoz = 1

export <- raw %>%
    select(c(varlist$var, "eid"))


export <- export %>%
    mutate(
        APOE_homoz = if_else(APOE > 1, 1, 0)
    )




fwrite(export,
    file = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UKBIOBANK/metadata/0-UKB_metadata_encoded.csv.gz",
)
