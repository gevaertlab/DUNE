suppressMessages({
    library(tidyverse)
    library(data.table)
})



varlist <- fread("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UPENN/metadata/0-variable_list.csv")

catg_cols <- varlist$var[varlist$task == "classification"]
reg_cols <- varlist$var[varlist$task == "regression"]
surv_cols <- c("death_delay", "death_event")




raw <- fread("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UPENN/metadata/UPENN-GBM_clinical_info_v1.0.csv", na.strings = "Not Available")

export <- raw %>%
    column_to_rownames("ID")

export$Gender <- ordered(export$Gender, levels = c("F", "M"))


export$IDH1 <- factor(export$IDH1)
export$IDH1 <- relevel(export$IDH1, "Wildtype")
export$IDH1 <- ordered(export$IDH1)
export$IDH1_bin <- ordered(export$IDH1 == "Wildtype", levels = c("FALSE", "TRUE"))
export$MGMT <- ordered(export$MGMT, levels=c("negative","positive"))
export$GTR_over90percent <- ordered(export$GTR_over90percent, levels=c("N","Y"))

export$death_event <- 1

# export <- export  %>% filter(!is.na(death_delay))

# ENCODING



encoded <- export %>% mutate_if(is.ordered, as.integer)
encoded[, catg_cols] <- encoded[, catg_cols] - 1

# NORMALISATION

normalized <- encoded
normalized[, reg_cols] <- scale(normalized[, reg_cols])
normalized <- normalized %>% rownames_to_column("eid")


fwrite(normalized,
    file = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UPENN/metadata/0-UPENN_metadata_encoded.csv"
)
