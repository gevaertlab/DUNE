suppressMessages({
    library(tidyverse)
    library(data.table)
})

set.seed(123)

var_path <- "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/TCGA/metadata/0-variable_list.csv"
raw_meta_path <- "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/TCGA/metadata/0-TCGA_metadata_encodedBCKP.csv"
list_patients <- list.files("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/TCGA/processed")



###
varlist <- fread(var_path)
catg_cols <- varlist$var[varlist$task == "classification" & varlist$keep_model == TRUE]
reg_cols <- varlist$var[varlist$task == "regression" & varlist$keep_model == TRUE]
surv_cols <- c("death_delay", "death_event")


raw <- fread(raw_meta_path, na.strings=c("", NA, "NA")) %>%
    distinct(eid, .keep_all = T) %>%
    # select(-c(case_id, survival_bin, grade_binary)) %>%
    filter(eid %in% list_patients)

export <- raw %>%
    column_to_rownames("eid")





export$Gender <- factor(export$Gender, levels= c(0,1), ordered=T)
# export$IDH1 <- factor(export$IDH1, ordered=T)
# export$IDH1 <- relevel(export$IDH1, 1)
export$IDH1 <- ordered(export$IDH1)
# export$IDH1_bin <- ordered(export$IDH1_bin, levels = c(0, 1))
# export$IDH2 <- factor(export$IDH2)
# export$IDH2 <- relevel(export$IDH2, 1)
export$IDH2 <- ordered(export$IDH2)
export$IDH2_bin <- ordered(export$IDH2_bin, levels = c(1, 2))
export$IDH_glob_bin <- export$IDH1_bin==2 | export$IDH2_bin==2
export$IDH_glob_bin <- ordered(export$IDH_glob_bin, levels = c(FALSE, TRUE))
export$grade <- ordered(as.factor(export$grade))
export$MGMT_promoter_status <- ordered(export$MGMT_promoter_status, levels = c("Unmethylated", "Methylated"))
export$CDKN2A_bin <- ordered(export$CDKN2A_bin, levels = c(1, 2))
export$Chr_7_gain_Chr_10_loss <- ordered(export$Chr_7_gain_Chr_10_loss, levels = c("No combined CNA", "Gain chr 
7 & loss chr 10"))
export$Chr_19_20_co_gain <- ordered(export$Chr_19_20_co_gain, levels = c("No chr 19/20 gain", "Gain chr 19/20"))
export$ATRX_bin <- ordered(export$ATRX_bin, levels = c(1, 2))
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
