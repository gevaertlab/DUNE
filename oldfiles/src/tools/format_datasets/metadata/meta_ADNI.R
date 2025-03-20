suppressMessages({
    library(tidyverse)
    library(data.table)
})



varlist <- fread("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/ADNI/metadata/0-variable_list.csv")

catg_cols <- varlist$var[varlist$task == "classification"]
reg_cols <- varlist$var[varlist$task == "regression"]


raw <- fread("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/ADNI/metadata/ADNI.csv")

export <- raw %>%
    arrange(eid, Visit ) %>%
    distinct(eid, .keep_all=T) %>%
    column_to_rownames("eid")


export$Group <- ordered(export$Group, levels = c("CN","MCI", "AD"))

export$dx <- fct_collapse(export$Group,
        normal = c("CN", "MCI"),
        alzheimer = c("AD")
    )
export$dx <- ordered(export$dx, levels = c("normal","alzheimer"))

export$Sex <- ordered(export$Sex, levels = c("F", "M"))
export$Age <- as.integer(export$Age)

# ENCODING
encoded <- export %>% mutate_if(is.ordered, as.integer)
encoded[, catg_cols] <- encoded[, catg_cols] - 1

# NORMALISATION

normalized <- encoded
normalized[, reg_cols] <- scale(normalized[, reg_cols])
normalized <- normalized %>% rownames_to_column("eid")


normalized$cohort <- sample(c("train", "test"), nrow(normalized), replace = TRUE, prob = c(0.8, 0.2))


fwrite(normalized,
    file = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/ADNI/metadata/0-ADNI_metadata_encoded.csv"
)



