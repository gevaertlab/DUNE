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


df <- data.table::fread(config$train_csv_path) %>%
# df <- data.table::fread(inn) %>%
    as_tibble() %>%
    select(-case)

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
    file = paste(paste(config$output_path, "/0-results.csv", sep = ""))
)

