suppressMessages({
    library(tidyverse)
    library(data.table)
})



varlist <- fread("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/SCHIZO/metadata/0-variable_list.csv")

catg_cols <- varlist$var[varlist$task == "classification"]
reg_cols <- varlist$var[varlist$task == "regression"]


raw <- fread("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/SCHIZO/metadata/SCHIZO_metadata.tsv", sep="\t")

export <- raw %>%
    column_to_rownames("eid")

export$sex <- ordered(export$sex, levels = c("female", "male"))
export$study <- ordered(export$study, levels = c("COBRE", "MCICShare"))
export$dx <- ordered(export$dx, levels = c("No_Known_Disorder", "Schizoaffective", "Schizophrenia_Broad", "Schizophrenia_Strict"))
export$age <- as.integer(export$age)

# ENCODING
encoded <- export %>% mutate_if(is.ordered, as.integer)
encoded[, catg_cols] <- encoded[, catg_cols] - 1

# NORMALISATION

normalized <- encoded
normalized[, reg_cols] <- scale(normalized[, reg_cols])
normalized <- normalized %>% rownames_to_column("eid")


normalized$cohort <- sample(c("train", "test"), nrow(normalized), replace = TRUE, prob = c(0.8, 0.2))


fwrite(normalized,
    file = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/SCHIZO/metadata/0-SCHIZO_metadata_encoded.csv"
)
