suppressMessages({
    library(tidyverse)
    library(data.table)
})

set.seed(123)

var_path <- "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/TCGA/metadata/0-variable_list.csv"
raw_meta_path <- "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/TCGA/metadata/TCGA_metadata.csv"
list_patients <- list.files("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/TCGA/images")



###
varlist <- fread(var_path)
catg_cols <- varlist$var[varlist$task == "classification" & varlist$keep_model == TRUE]
reg_cols <- varlist$var[varlist$task == "regression" & varlist$keep_model == TRUE]
surv_cols <- c("death_delay", "death_event")


raw <- fread(raw_meta_path) %>%
    distinct(eid, .keep_all = T) %>%
    select(-c(case_id, survival_bin, grade_binary)) %>%
    filter(eid %in% list_patients)

export <- raw %>%
    column_to_rownames("eid")





export$Gender <- factor(export$Gender, levels= c("Male","Female"), ordered=T)
export$IDH1 <- factor(export$IDH1)
export$IDH1 <- relevel(export$IDH1, "WT")
export$IDH1 <- ordered(export$IDH1)
export$IDH1_bin <- ordered(export$IDH1_bin, levels = c("FALSE", "TRUE"))
export$IDH2 <- factor(export$IDH2)
export$IDH2 <- relevel(export$IDH2, "WT")
export$IDH2 <- ordered(export$IDH2)
export$IDH2_bin <- ordered(export$IDH2_bin, levels = c("FALSE", "TRUE"))
export$IDH_glob_bin <- export$IDH1_bin=="TRUE" | export$IDH2_bin=="TRUE"
export$IDH_glob_bin <- ordered(export$IDH_glob_bin, levels = c("FALSE", "TRUE"))
export$grade <- ordered(as.factor(export$grade))
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




normalized$cohort <- sample(c("train", "test"), nrow(normalized), replace = TRUE, prob = c(0.8, 0.2))

fwrite(normalized,
    file = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/TCGA/metadata/0-TCGA_metadata_encoded.csv"
)
