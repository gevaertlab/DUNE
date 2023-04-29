suppressMessages({
    library(tidyverse)
    library(finalfit)
    library(ggthemes)
})


INPUT_FILE <- "/home/tbarba/projects/MultiModalBrainSurvival/outputs/tumor_crop.csv"

var_levels <- c("death","IDH_glob_bin", "Gender", "grade")
AE_levels <- c("AE_UCSF_segm", "AE_TCGA_segm", "AE_UPENN_segm")
feat_levels <- c("whole_brain","tumor", "combined", "radiomics")
cols <- c("#1e28af", "#911212","#5a1074", "#0c6625")

df <- read_csv(INPUT_FILE) %>%
    mutate(
        AE = factor(AE, levels = AE_levels),
        var = factor(var, levels = var_levels),
        features = factor(features, levels = feat_levels),
        type_AE =factor(str_split(AE, "_", simplify = TRUE)[, 1], levels=c("AE")),
        dataset = factor(str_split(AE, "_", simplify = TRUE)[, 2], levels = c("UCSF", "TCGA", "UPENN"))     )%>%
        filter(var != "grade")


df %>%
    filter(features %in% c("whole_brain","tumor","combined", "radiomics")) %>%
    # filter(type_AE != "VAE") %>%
    ggplot(
        aes(x = var, y = performance, fill=features)
    ) +
    geom_col(position = "dodge", col = "black", alpha=0.8) +
    # geom_point(size = 2) +
        # geom_line(size = 1) +
        facet_wrap(~dataset, ncol=3) +
        scale_y_continuous(limits = c(0,1), breaks=seq(0, 1, 0.1)) +
    theme_stata() +
    scale_fill_manual(values=cols) 

# df %>%
#     filter(features %in% c("combined", "radiomics")) %>%
#     ggplot(
#         aes(x = var, y = performance, fill = features, alpha = type_AE)
#     ) +
#     geom_col(position = "dodge", col = "black") +
#     # geom_point(size = 2) +
#     # geom_line(size = 1) +
#     facet_wrap(~dataset, ncol = 3) +
#     scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1)) +
#     theme_stata() +
#     scale_fill_manual(values = cols) +
#     scale_alpha_discrete(range = c(1, 0.5))
