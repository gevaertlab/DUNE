library("tidyverse")
library("gt")

MODELS_DIR = "/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/finetuning"

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


create_cat_table <- function(df, cohort) {

    models <- c(paste("6b_4f_", cohort, sep = ""), paste("6b_4f_", cohort, "_segm", sep = ""))

    df <- df %>%
        filter(model_ %in% models) %>%
        pivot_wider(id_cols = c("var"), values_from = c("performance"), names_from = model_, names_sort = T) %>%
        left_join(variances, by = "var") %>%
        relocate(variance, .after = "var") %>%
        relocate(nb_classes, .after = "variance") %>%
        relocate(missing_rate, .after = "nb_classes") %>%
        arrange(desc(variance), .by_group = T)

    cat_table <- df %>% gt() %>%
        cols_align(columns = matches("variance|nb_classes|missing_rate|6b_"), align = "center") %>%
        summary_rows(
            groups = F,
            columns = matches("6b_"),
            fns = list(avg = ~ mean(., na.rm = T))
        ) %>%
        grand_summary_rows(
            columns = matches("6b_"),
            fns = list("GLOBAL AVG" = ~ mean(., na.rm = T))
        ) %>%
        data_color(
            columns = matches("6b_"),
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



## CONCAT
segm_models <- c("6b_4f_TCGA_segm", "6b_4f_UCSF_segm", "6b_4f_UPENN_segm", "6b_4f_REMB_segm")

multi <- tibble(data.table::fread("/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/finetuning/0-synth_multi.csv")) %>%
    mutate(performance = round(performance, 4), segm=if_else(model_ %in% segm_models, TRUE, FALSE))



variances <- multi %>%
    select(var, variance, missing_rate, model, restored_model, num_classes) %>%
    mutate(variance = round(variance, 2)) %>%
    distinct(var, .keep_all = T) %>%
    rename("nb_classes" = "num_classes")


