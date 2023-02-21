suppressMessages({
    library(tidyverse)
    library(ggpubr)
    library(data.table)
    library("gt")
})


gt_theme_538 <- function(data, ...) {
    data %>%
        opt_all_caps() %>%
        opt_table_font(font = list(google_font("Chivo"), default_fonts())) %>%
        tab_style(
            style = cell_borders(sides = "bottom", color = "transparent", weight = px(2)),
            locations = cells_body(columns = TRUE, rows = nrow(data$`_data`))
        ) %>%
        tab_options(
            column_labels.background.color = "white",
            table.border.top.width = px(3),
            table.border.top.color = "transparent",
            table.border.bottom.color = "transparent",
            table.border.bottom.width = px(3),
            column_labels.border.top.width = px(3),
            column_labels.border.top.color = "transparent",
            column_labels.border.bottom.width = px(3),
            column_labels.border.bottom.color = "black",
            data_row.padding = px(3),
            source_notes.font.size = 12,
            table.font.size = 16,
            heading.align = "left",
            ...
        )
}

create_comp_table <- function(df, comparison_cols, subgroup) {
    keep = c("group", "var", "task", "num_classes", "variance", comparison_cols)
    ResultTable <- df %>%
        filter(group %in% subgroup) %>%
        select(keep) %>%
        gt(groupname_col = "group") %>%
        cols_align(columns = matches("variance|nb_classes|accuracy|f1_weighted|Ridge|RandomForest"), align = "center") %>%
        summary_rows(
            groups = TRUE,
            columns = comparison_cols,
            fns = list(avg = ~ mean(., na.rm = T))
        ) %>%
        grand_summary_rows(
            columns = comparison_cols,
            fns = list("GLOBAL AVG" = ~ mean(., na.rm = T))
        ) %>%
        tab_spanner(
            label = "Performance",
            columns = comparison_cols
        ) %>%
        data_color(
            columns = comparison_cols,
            colors = scales::col_numeric(
                c("#0a4c6a", "#73bfe2", "#cfe8f3", "#fff2cf", "#fdd870", "#ca0000"),
                domain = c(0, 1, NA), na.color = "#444444"
            )
        ) %>%
        data_color(
            columns = matches("variance"),
            colors = scales::col_numeric(c("#117926", "#ca0000"),
                domain = range(0, 1.7)
            )
        ) %>%
        gt_theme_538()
    return(ResultTable)

}




Ridge_acc <- tibble(fread("outputs/UNet/pretraining/UNet_6b_4f_UKfull/multivariate/Ridge_Acc.csv"))
Ridge_f1 <- tibble(fread("outputs/UNet/pretraining/UNet_6b_4f_UKfull/multivariate/Ridge_F1.csv"))
RandFor_f1 <- tibble(fread("outputs/UNet/pretraining/UNet_6b_4f_UKfull/multivariate/RandFor_F1.csv"))


# D'abord comparer F1 et Accuracy


names(Ridge_acc)[9] <- "accuracy"
names(Ridge_f1)[9] <- "f1_weighted"

Ridge_tables <- Ridge_acc %>%
    column_to_rownames("var") %>%
    left_join(Ridge_f1)



# Groups

volumetry <- c("acquisition", "volumetry")
global <- c("global", "recruitment")
vascular <- c("carotid", "arterial_stiffness", "blood_pressure", "diabetes", "biology")
lifestyle <- c("alcohol", "smoking", "sleep", "multimedia")
diet <- "diet"
conditions <- c("conditions")
psy <- c("psy")
genes <- c("Alzheimer", "Multiple Sclerosis", "Parkinson", "Bipolar", "Schizo")


# Acc vs F1 genet
perf_cols <- c("accuracy", "f1_weighted")

create_comp_table(Ridge_tables, perf_cols, genes)



# Ridge vs Random forest

Ridge_f1 <- tibble(fread("outputs/UNet/pretraining/UNet_6b_4f_UKfull/multivariate/Ridge_F1.csv"))
RandFor_f1 <- tibble(fread("outputs/UNet/pretraining/UNet_6b_4f_UKfull/multivariate/RandFor_F1.csv"))
names(Ridge_f1)[9] <- "Ridge"
names(RandFor_f1)[9] <- "RandomForest"

F1_models <- Ridge_f1 %>%
    column_to_rownames("var") %>%
    left_join(RandFor_f1)

comp_cols <- c("Ridge", "RandomForest")




volumetry <- c("acquisition", "volumetry")
global <- c("global", "recruitment")
vascular <- c("carotid", "arterial_stiffness", "blood_pressure", "diabetes", "biology")
lifestyle <- c("alcohol", "smoking", "sleep", "multimedia")
diet <- "diet"
conditions <- c("conditions")
psy <- c("psy")
genes <- c("Alzheimer", "Multiple Sclerosis", "Parkinson", "Bipolar", "Schizo")



create_comp_table(F1_models, comp_cols, psy)




# Survival
# USCF

UCSF_ridge <- tibble(fread("/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/finetuning/6b_4f_UCSF/multivariate/Ridge.csv"))

UCSF_random <- tibble(fread("/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/finetuning/6b_4f_UCSF/multivariate/0-multivariate.csv")) 

names(UCSF_ridge)[9] <- "Ridge"
names(UCSF_ridge)[8] <- "IBS_Ridge"
names(UCSF_random)[9] <- "RandomForest"
names(UCSF_random)[8] <- "IBS_Random"
comp_cols <- c("Ridge", "RandomForest")


UCSF_tables <- UCSF_ridge %>%
    left_join(UCSF_random)


create_comp_table(UCSF_tables, comp_cols, UCSF_tables$group)


# TCGA

TCGA_ridge <- tibble(fread("/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/final/6b_4f_TCGA/multivariate/Ridge.csv"))

TCGA_random <- tibble(fread("/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/final/6b_4f_TCGA/multivariate/0-multivariate.csv"))

names(TCGA_ridge)[9] <- "Ridge"
names(TCGA_ridge)[8] <- "IBS_Ridge"
names(TCGA_random)[9] <- "RandomForest"
names(TCGA_random)[8] <- "IBS_Random"
comp_cols <- c("Ridge", "RandomForest")


TCGA_tables <- TCGA_ridge %>%
    left_join(TCGA_random)


create_comp_table(TCGA_tables, comp_cols, TCGA_tables$group)