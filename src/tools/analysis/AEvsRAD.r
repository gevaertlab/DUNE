suppressMessages({
    library(tidyverse)
    library(finalfit)
    library(ggthemes)
})


INPUT_FILE <- "/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/finetuning/AEvsRAD.csv"


df <- read_csv(INPUT_FILE) 


selected <- c("death","IDH_glob_bin","grade")

df %>%
    mutate(var = factor(var, levels = selected)) %>%
    mutate(features = factor(features, levels = c("features","radiomics","combined"))) %>%
    ggplot(
        aes(x = var, y = performance, group = features, col=features)
    ) +
    # geom_col(position = "dodge", col = "black") +
    geom_point(size = 2) +
        geom_line(size = 1) +
        facet_grid(~AE) +
        ylim(0,1) +
    theme_stata()
