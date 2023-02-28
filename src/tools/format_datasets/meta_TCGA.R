suppressMessages({
    library(tidyverse)
    library(data.table)
})



varlist <- fread("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/TCGA/metadata/0-variable_list.csv")

catg_cols <- varlist$var[varlist$task == "classification"]
reg_cols <- varlist$var[varlist$task == "regression"]
surv_cols <- c("death_delay", "death_event")




raw <- fread("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/TCGA/metadata/TCGA_metadata.csv") %>%
    distinct(eid, .keep_all = T) %>%
    select(-c(case_id, survival_bin, grade_binary))

export <- raw %>%
    column_to_rownames("eid")

export$IDH1 <- factor(export$IDH1)
export$IDH1 <- relevel(export$IDH1, "WT")
export$IDH1 <- ordered(export$IDH1)
export$grade <- ordered(as.factor(export$grade))
export$IDH1_bin <- ordered(export$IDH1_bin, levels = c("FALSE", "TRUE"))
export$CDKN2A_bin <- ordered(export$CDKN2A_bin, levels = c("FALSE", "TRUE"))
export$ATRX_bin <- ordered(export$ATRX_bin, levels = c("FALSE", "TRUE"))
# export <- export  %>% filter(!is.na(death_delay))

# ENCODING

encoded <- export %>% mutate_if(is.ordered, as.integer)
encoded[, catg_cols] <- encoded[, catg_cols] - 1

# NORMALISATION

normalized <- encoded
normalized[, reg_cols] <- scale(normalized[, reg_cols])
normalized <- normalized %>% rownames_to_column("eid")


fwrite(normalized,
    file = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/TCGA/metadata/0-TCGA_metadata_encoded.csv"
)
