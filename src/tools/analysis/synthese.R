library("tidyverse")
library("gt")

MODELS_DIR = "/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/pretraining"

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

create_table <- function(csv_file, metric) {

    csv_path = paste(MODELS_DIR, csv_file, sep = "/")
    summary <- tibble(data.table::fread(csv_path)) %>%
        mutate(
            group = as.factor(group),
            variance = round(variance, 2),
            model = str_replace(model, "UNet_5b_8f_UKfull", "5B_8F"),
            model = str_replace(model, "UNet_5b_4f_UKfull", "5B_4F"),
            model = str_replace(model, "UNet_6b_8f_UKfull", "6B_8F"),
            model = str_replace(model, "UNet_6b_4f_UKfull", "6B_4F"),
            model = factor(model, levels = c("5B_8F", "5B_4F", "6B_8F","6B_4F"))
        ) %>%
        rename(res = metric)

    list_mod = unique(summary$model)

    variances <- summary %>%
        select(var, variance, num_classes) %>%
        distinct(var, .keep_all = T)

    maximum = max(summary$res, na.rm=T) + 0.01

    Table <- summary %>%
        # filter(!var %in% c("sleep", "weight")) %>%
        mutate(res = round(res, 3), variance = round(variance, 1)) %>%
        arrange(model) %>%
        pivot_wider(id_cols = c("var", "task", "group"), values_from = res, names_from = model) %>%
        left_join(variances, by="var") %>% group_by(group) %>%
        relocate(variance, .after="group") %>%
        relocate(num_classes, .before="variance") %>%
        arrange(desc(variance), .by_group = T)
    
    mod_cols = (dim(Table)[2] - length(list_mod) +1):dim(Table)[2]

    Table <- Table %>% gt(groupname_col = "group") %>%
        summary_rows(
            groups = TRUE,
            columns = list_mod,
            fns = list(avg = ~ mean(., na.rm=T))
        ) %>%
        grand_summary_rows(
            columns = list_mod,
            fns = list("GLOBAL AVG" = ~ mean(., na.rm=T))
        ) %>%
        tab_spanner(
            label = "Models",
            columns = mod_cols
        ) %>%
        data_color(
            columns = list_mod,
            colors = scales::col_numeric(
                c("#0a4c6a", "#73bfe2", "#cfe8f3", "#fff2cf", "#fdd870", "#ca0000"),
                domain = c(0, maximum, NA), na.color = "#444444"
            )
        ) %>%
        gtable_add_space()
        gt_theme_538()


        return(Table)
}

# univ_table <- create_table("univ_summaryNORM.csv", metric = "proportion_sig")
# multi_table <- create_table("multi_summaryNORM.csv", metric = "performance")


# gtsave(univ_table, "results/UNIV.html")
# gtsave(multi_table, "results/MULTI_RIDGE.html")


## CONCAT

multi <- tibble(data.table::fread("/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/pretraining/0-synth_multi.csv")) %>%
    mutate(performance = round(performance, 2))

univ <- tibble(data.table::fread("/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/pretraining/univ_summaryNORM.csv")) %>%
    select(-num_classes) %>%
    select(var, model, proportion_sig) %>%
    mutate(proportion_sig = round(proportion_sig, 2))


variances <- multi %>%
    select(var, variance, num_classes) %>%
    mutate(variance = round(variance, 2)) %>%
    distinct(var, .keep_all = T) %>%
    rename("variance" = "variance", "nb_classes" = "num_classes")



concat <- multi %>%
    mutate(model = recode(model, "UNet_5b_8f_UKfull"="5B_8F", "UNet_5b_4f_UKfull"="5B_4F", "UNet_6b_8f_UKfull"="6B_8F","UNet_6b_4f_UKfull"="6B_4F")) %>%
    left_join(univ, by = c("model", "var")) %>%
    rename("U" = "proportion_sig", "M" = "performance") %>%
    pivot_wider(id_cols = c("var", "task", "group"), values_from = c("U", "M"), names_from = model, names_sort = T) %>%
    left_join(variances, by = "var") %>%
    relocate(variance, .after = "group") %>%
    relocate(nb_classes, .after = "variance") %>%
    group_by(group) %>%
    arrange(desc(variance), .by_group = T)


create_cat_table <- function(df , subgroup) {
    cat_table <- df %>%
        filter(var %in% subgroup) %>%
        select(-matches("U_")) %>%
        gt(groupname_col = "group") %>%
        row_group_order(groups = c("volumetry","vascular", "psy","lifestyle", "SNPs")) %>%
        cols_align(columns = matches("variance|nb_classes|U_|M_"), align = "center") %>%
        summary_rows(
            groups = TRUE,
            columns = matches("U_|M_"),
            fns = list(avg = ~ mean(., na.rm = T))
        ) %>%
        grand_summary_rows(
            columns = matches("U_|M_"),
            fns = list("GLOBAL AVG" = ~ mean(., na.rm = T))
        ) %>%
        # tab_spanner(
        #     label = "Univariate : proportion signif",
        #     columns = matches("U_"),
        # ) %>%
        tab_spanner(
            label = "Multivar : prediction (r2 / accuracy)",
            columns = matches("M_")
        ) %>%
        data_color(
            columns = matches("U_|M_"),
            colors = scales::col_numeric(
                c("#0a4c6a", "#73bfe2", "#cfe8f3", "#fff2cf", "#fdd870", "#ca0000"),
                domain = range(0, 1)
            )
        ) %>%
        data_color(
            columns = matches("variance"),
            colors = scales::col_numeric(c("#117926", "#ca0000"),
                domain = range(0, 1.7)
            )
        ) %>%
        gt_theme_538()

        return(cat_table)
}

## EXPORT


volumetry <- c("acquisition", "volumetry")
global <- c("global", "recruitment")
vascular <- c("carotid", "arterial_stiffness", "blood_pressure", "diabetes", "biology")
lifestyle <- c("alcohol", "smoking", "sleep", "multimedia", "lifestyle")
diet <- "diet"
conditions <- c("conditions")
psy <- c("psy")
genes <- c("Alzheimer", "Multiple Sclerosis", "Parkinson", "Bipolar", "Schizo")



retained <- c(
"Volume_of_brain_grey_white_matter",
"Volume_of_white_matter", 
"Volume_of_grey_matter", 
#  "Volume_of_thalamus_right",
#  "Volume_of_thalamus_left",
#  "Volume_of_pallidum_right",
#  "Volume_of_pallidum_left",
#  "Volume_of_putamen_right",
#  "Volume_of_putamen_left",
"Gender",
#  "Height",
"Plays_computer_games",
"Daytime_dozing_or_sleeping",
"Sleeplessness_or_insomnia",
"Smoking_status", "Salt_added_to_food",
"Seen_doctor_GP_for_nerves_anxiety_tension_or_depression",
"Suffer_from_nerves",
"Seen_a_psychiatrist_for_nerves_anxiety_tension_or_depression",
"Tense_or_highly_strung",
"Risk_taking",
"Irritability", 
# "Qualifications",
"Ever_depressed_for_a_whole_week",
"Vascular_heart_problems_diagnosed_by_doctor",
"APOE", "CR1","TRANK1", "HLA-DRB1", "DISC1", "SNCA"
)

genes <- c(
    "APOE", "CR1","TRANK1", "HLA-DRB1", "DISC1", "SNCA"
)

# concat$group <- factor(concat$group, levels = c(volumetry, global, vascular, lifestyle, diet, conditions, psy))


concat2 <- concat %>%
    mutate(
        group = fct_collapse(group,
            volumetry = "volumetry",
            psy = "psy",
            vascular = c("global","conditions", "smoking"),
            lifestyle = c("multimedia", "diet", "sleep", "lifestyle"),
            SNPs = c("Alzheimer", "Bipolar", "Parkinson", "Multiple Sclerosis","Schizo"),
        other_level = "other"),
        group = factor(group, levels = c("volumetry", "vascular", "psy", "lifestyle", "SNPs", "other"))
    )

gtsave(create_cat_table(concat2, retained), "results/1-synth.html")
gtsave(create_cat_table(concat2, genes), "results/1-newgenes.html")