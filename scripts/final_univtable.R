library(tidyverse)
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
            group = factor(group, levels = c("volumetry", "global", "function", "agression")),
            model = factor(model, levels = c("UNet_5b_8f_UKfull", "UNet_5b_4f_UKfull", "UNet_6b_8f_UKfull", "UNet_6b_4f_UKfull")),
            model = recode_factor(model,
                "UNet_5b_8f_UKfull" = "5B_8F",
                "UNet_5b_4f_UKfull" = "5B_4F",
                "UNet_6b_8f_UKfull" = "6B_8F",
                "UNet_6b_4f_UKfull" = "6B_4F"
            )
        ) %>%
        rename(res = metric)

    maximum = max(summary$res) + 0.01

    Table <- summary %>%
        filter(!var %in% c("sleep", "weight")) %>%
        mutate(res = round(res, 3)) %>%
        arrange(model) %>%
        pivot_wider(id_cols = c("var", "task", "group"), values_from = res, names_from = model) %>%
        gt(groupname_col = "group") %>%
        summary_rows(
            groups = TRUE,
            columns = c("5B_8F", "5B_4F", "6B_8F", "6B_4F"),
            fns = list(avg = ~ mean(., na.rm=T))
        ) %>%
        grand_summary_rows(
            columns = c("5B_8F", "5B_4F", "6B_8F", "6B_4F"),
            fns = list("GLOBAL AVG" = ~ mean(., na.rm=T))
        ) %>%
        tab_spanner(
            label = "Models",
            columns = 3:7
        ) %>%
        data_color(
            columns = c("5B_8F", "5B_4F", "6B_8F", "6B_4F"),
            colors = scales::col_numeric(
                c("#0a4c6a", "#73bfe2", "#cfe8f3", "#fff2cf", "#fdd870", "#ca0000"),
                domain = range(0, maximum)
            )
        ) %>%
        gt_theme_538()


        return(Table)
}

univ_table <- create_table("univ_summary.csv", metric = "proportion_sig")
multi_table <- create_table("multi_summary.csv", metric = "performance")
# univ_table
# multi_table

univ <- tibble(data.table::fread("/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/pretraining/univ_summary.csv")) 
# multi <- tibble(data.table::fread("/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/pretraining/multi_summary.csv")) 

# concat <- left_join(univ, multi) %>%
#     mutate(
#         group = factor(group, levels = c("volumetry", "global", "function", "agression")),
#         model = factor(model, levels = c("UNet_5b_8f_UKfull", "UNet_5b_4f_UKfull", "UNet_6b_8f_UKfull", "UNet_6b_4f_UKfull")),
#         model = recode_factor(model,
#             "UNet_5b_8f_UKfull" = "5B_8F",
#             "UNet_5b_4f_UKfull" = "5B_4F",
#             "UNet_6b_8f_UKfull" = "6B_8F",
#             "UNet_6b_4f_UKfull" = "6B_4F"
#         )
#     ) %>%
#     mutate(
#         proportion_sig = round(proportion_sig, 3),
#         performance = round(performance, 3),
#     ) %>%
#         filter(!(var %in% c("weight","sleep")))
    
    
    
    
    
# cat_table <- concat %>%
#     rename("univ" = "proportion_sig", "multi" = "performance") %>%
#     pivot_wider(id_cols = c("var", "task", "group"), values_from = c("univ", "multi"), names_from = model, names_sort = T) %>%
#     gt(groupname_col = "group") %>%
#         cols_align(columns = matches("univ|multi"), align="center") %>%
#         summary_rows(
#             groups = TRUE,
#             columns = matches("univ|multi"),
#             fns = list(avg = ~ mean(., na.rm=T))
#         ) %>%
#         grand_summary_rows(
#             columns = matches("univ|multi"),
#             fns = list("GLOBAL AVG" = ~ mean(., na.rm=T))
#         ) %>%
#         tab_spanner(
#             label = "Univariate",
#             columns = 3:7,
#         ) %>%
#         tab_spanner(
#             label = "Multivar. prediction",
#             columns = 8:11
#         ) %>%
#         data_color(
#             columns = matches("univ|multi"),
#             colors = scales::col_numeric(
#                 c("#0a4c6a", "#73bfe2", "#cfe8f3", "#fff2cf", "#fdd870", "#ca0000"),
#                 domain = range(0, 1)
#             )
#         ) %>%
#         gt_theme_538() 

# cat_table

gtsave(univ_table, "test.html")
