suppressMessages({
    library(tidyverse)
    library(data.table)
})

set.seed(123)

var_path <- "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UPENN/metadata/0-variable_list.csv"
raw_meta_path <- "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UPENN/metadata/UPENN_metadata.csv"
list_patients <- list.files("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UPENN/images")





varlist <- fread(var_path)
catg_cols <- varlist$var[varlist$task == "classification"]
reg_cols <- varlist$var[varlist$task == "regression"]
surv_cols <- c("death_delay", "death_event")


raw <- fread(raw_meta_path, na.strings = c("Not Available", "NOS/NEC")) %>%
    filter(ID %in% list_patients)

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



# ENCODING
encoded <- export %>% mutate_if(is.ordered, as.integer)
encoded[, catg_cols] <- encoded[, catg_cols] - 1

# NORMALISATION

normalized <- encoded
normalized[, reg_cols] <- scale(normalized[, reg_cols])
normalized <- normalized %>% rownames_to_column("eid")


normalized$cohort <- sample(c("train", "test"), nrow(normalized), replace = TRUE, prob = c(0.8, 0.2))

fwrite(normalized,
    file = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UPENN/metadata/0-UPENN_metadata_encoded.csv"
)
