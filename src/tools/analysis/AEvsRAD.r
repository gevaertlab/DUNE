suppressMessages({
    library(tidyverse)
    library(finalfit)
    library(ggthemes)
})


INPUT_FILE <- "/home/tbarba/projects/MultiModalBrainSurvival/outputs/AEvsRAD3.csv"

var_levels <- c("death","IDH_glob_bin", "Gender", "grade")
AE_levels <- c("UNet_UCSF_segm", "VAE_UCSF_segm", "UNet_UPENN_segm", "VAE_UPENN_segm", "UNet_TCGA_segm", "VAE_TCGA_segm")
feat_levels <- c("features", "radiomics", "combined")
cols <- c("#af1e1e", "#10a088", "#2113a1")

df <- read_csv(INPUT_FILE) %>%
    mutate(
        AE = factor(AE, levels = AE_levels),
        var = factor(var, levels = var_levels),
        features = factor(features, levels = feat_levels),
        type_AE =factor(str_split(AE, "_", simplify = TRUE)[, 1], levels=c("UNet","VAE")),
        dataset = factor(str_split(AE, "_", simplify = TRUE)[, 2], levels = c("UCSF", "TCGA", "UPENN"))     )%>%
        filter(var != "grade")


df %>%
    # filter(features != "combined") %>%
    filter(!(features == "combined" & type_AE == "UNet")) %>%
    # filter(type_AE != "VAE") %>%
    ggplot(
        aes(x = var, y = performance, fill=features, alpha=type_AE)
    ) +
    geom_col(position = "dodge", col = "black") +
    # geom_point(size = 2) +
        # geom_line(size = 1) +
        facet_wrap(~dataset, ncol=3) +
        scale_y_continuous(limits = c(0,1), breaks=seq(0, 1, 0.1)) +
    theme_stata() +
    scale_fill_manual(values=cols) +
    scale_alpha_discrete(range=c(1,0.5))


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
