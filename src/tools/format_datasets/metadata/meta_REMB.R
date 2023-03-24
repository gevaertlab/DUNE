suppressMessages({
    library(tidyverse)
    library(data.table)
})



varlist <- fread("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/REMBRANDT/metadata/0-variable_list.csv")

catg_cols <- varlist$var[varlist$task == "classification"]
reg_cols <- varlist$var[varlist$task == "regression"]
surv_cols <- c("death_delay", "death_event")




raw <- fread("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/REMBRANDT/metadata/REMBRANDT_metadata.csv", na.strings = c("",NA)) %>%
    distinct(eid, .keep_all = T)


export <- raw %>%
    column_to_rownames("eid")

export$age <- ordered(export$age, levels = c("10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "85-89"))
export$sex <- ordered(export$sex, levels=c("F","M"))
export$death_event <- 1
export$disease <- ordered(export$disease, levels=c("ASTROCYTOMA","GBM","MIXED", "OLIGODENDROGLIOMA"))
export$grade <- ordered(export$grade, levels=c("II","III","IV"))

# ENCODING

encoded <- export %>% mutate_if(is.ordered, as.integer)
encoded[, catg_cols] <- encoded[, catg_cols] - 1

# NORMALISATION

normalized <- encoded
normalized[, reg_cols] <- scale(normalized[, reg_cols])
normalized <- normalized %>% rownames_to_column("eid")


fwrite(normalized,
    file = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/REMBRANDT/metadata/0-REMB_metadata_encoded.csv"
)
