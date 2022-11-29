library("tidyverse")
library("rjson")
library("umap")
library("cowplot")
library("optparse")

grouped_wilcox <- function(variable, df = df) {
    df <- df %>%
        summarize(
            across(
                .cols = `0`:ncol(df),
                ~ wilcox.test(. ~ !!sym(variable))$p.value
            )
        )
    ratio <- df %>%
        mutate(
            countsign = rowSums(. * ncol(df) < 0.05, na.rm = T),
            total = ncol(df),
            ratio = countsign / total
        ) %>%
        pull(ratio)

    return(ratio)
}

grouped_spearman <- function(variable, df = df) {
    df <- df %>%
        summarize(
            across(
                .cols = `0`:ncol(df),
                ~ cor.test(x = ., y = !!sym(variable), method = "spearman")$p.value
            )
        )
    ratio <- df %>%
        mutate(
            countsign = rowSums(. * ncol(df) < 0.05, na.rm = T),
            total = ncol(df),
            ratio = countsign / total
        ) %>%
        pull(ratio)

    return(ratio)
}

draw_umap <- function(umap=umap, labels=labels, var = var) {
    p <- data.frame(
        x = umap$layout[, 1],
        y = umap$layout[, 2],
        var = labels[, var]
    ) %>%
        ggplot(aes_string("x", "y", col = var)) +
            geom_point() +
            theme_linedraw()
        
    ggsave(p, filename = paste(config$output_path, "/", var, ".png", sep=""))
}


option_list <- list(
    make_option(c("-c", "--config"),
        type = "character", default = NULL,
        help = "config file path", metavar = "character"
    )
)
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)
config = fromJSON(file = opt$config)

#####
DATA = paste(config$model_path, config$train_csv_path, sep = "")
config$output_path = paste(config$model_path, config$output_path, sep = "")
TAF::mkdir(config$output_path)
    
df <- data.table::fread(DATA) %>%
    as_tibble() %>%
    select(-case) %>%
    mutate(
        alcohol_status = ordered(alcohol_status, c("Never", "Previous", "Current")),
        alcohol_freq = ordered(alcohol_freq, c("Never", "Special occasions only", "One to three times a month", "Once or twice a week", "Three or four times a week", "Daily or almost daily")),
        smoking = ordered(smoking, c("Never", "Previous", "Current", "Prefer not to answer")),
        depression = ordered(depression, c("No", "Yes"))
    ) %>%
    mutate(
        sex = as.factor(sex),
        alcohol_status = as.numeric(alcohol_status),
        alcohol_freq = as.numeric(alcohol_freq),
        smoking = as.numeric(smoking),
    )

umap <- umap(df %>% select(num_range("", 0:20000)))
labels <- df %>%
    select(!num_range("", 0:20000)) %>%
    mutate(across(where(~ all(unique(.[!is.na(.)]) %in% c("0", "1"))), as.factor))


results = list()
for (var in colnames(labels)) {
    isFactor <- is.factor(labels %>% select(var) %>% pull())
    if (isFactor) {
        res = grouped_wilcox(variable = var, df = df)
    } else {
        res = grouped_spearman(variable = var, df = df)
    }
    names(res) = var
    results = append(results, res)
    draw_umap(umap=umap, labels=labels, var = var)
}
write_csv(
    as.data.frame(results),
    file = paste(paste(config$output_path, "/0-univariate_results.csv", sep = ""))
)

