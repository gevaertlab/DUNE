library("tidyverse")
library("rjson")
library("umap")
library("cowplot")
library("optparse")
library("caret")
library("cowplot")

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

grouped_anova <- function(variable, df) {
    df <- df %>%
        summarize(
            across(
                .cols = `0`:ncol(df),
                ~ anova(lm(. ~ !!sym(variable)))[["Pr(>F)"]][1]
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

draw_umap <- function(umap, labels, var_name, var_class) {
    var = labels %>% pull(var_name)
    p <- data.frame(
        x = umap$layout[, 1],
        y = umap$layout[, 2],
        var = var  ) %>%
        ggplot(aes_string("x", "y", col = var)) +
            geom_point() +
            theme_linedraw() +
            ggtitle(var_name)
    return(p)
}


# PARSING
option_list <- list(
    make_option(c("-c", "--config"),
        type = "character", default = NULL,
        help = "config file path", metavar = "character"
    )
)
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)
config = fromJSON(file = opt$config)


# IMPORTS
METADATA <- paste(config$model_path, config$metadata_path, sep = "")
VARIABLES <- paste(config$model_path, config$variables_path, sep = "")
FEATURES <- paste(config$model_path, config$feature_path, sep = "")
OUTPUT_PATH <- paste(config$model_path, output_path, sep = "")
UMAP <- as.logical(config$umap)

TAF::mkdir(OUTPUT_PATH)


features = bind_rows(
    data.table::fread(paste(FEATURES, "concat_train.csv", sep=""), header=T),
    data.table::fread(paste(FEATURES, "concat_val.csv", sep=""), header=T)  
) %>% as_tibble() %>%
    rename(eid = V1) %>%
    mutate(eid = substr(eid, 1, 7))

metadata <- data.table::fread(METADATA, stringsAsFactors = T) %>%
    mutate(eid = as.character(eid)) %>%
    as_tibble()
    
variables <- data.table::fread(VARIABLES) %>%
    as_tibble() %>%
    filter(newname %in% names(metadata))

factors <- variables$newname[variables$task == "classification"]
metadata <- metadata %>%
    mutate_at(factors, as.factor)
    
# MISSING METADATA IMPUTATION (KNN)
imputer <- preProcess(as.data.frame(metadata), method = c("knnImpute"), k = 120, knnSummary = mean)
imputed_metadata <- predict(imputer, metadata, na.action=pass)


# FUSED DF
df <- metadata %>% right_join(features, by = ("eid"))


# ANALYSIS
labels <- df %>%
    select(!num_range("", 0:50000)) %>%
    mutate(across(where(~ all(unique(.[!is.na(.)]) %in% c("0", "1"))), as.factor))

if (UMAP) {
    Umap <- umap(df %>% select(num_range("", 0:50000)))
}



results <- tibble(variables) 
results$res <- numeric(nrow(results))
plot_list <- list()
i <- 1
# for (i in 1:length(results$newname)) {
for (i in 1:3) {
    var = results$newname[i]
    task = results$task[i]
    res = NA
    print(var)

    if (task == "regression") {
        res = grouped_spearman(variable = var, df = df)
    } else {
        if (length(levels(df %>% pull(var)) < 3)) {
            res = grouped_wilcox(variable = var, df = df)
        } else {
            res = grouped_anova(variable = var, df = df)
        }
    }

    results[results$newname == var, "proportion_sig"] = res

    if (UMAP) {
        plot_list[[i]] = draw_umap(umap = Umap, labels = labels, var_name = var, var_class = task)
    }
}


if (UMAP) {
    pl <- plot_grid(plotlist = plot_list)
    ggsave(paste(OUTPUT_PATH, "/umaps.pdf", sep = ""), pl)
}


write_csv(
    results,
    file = paste(OUTPUT_PATH, "/0-univariate_results.csv", sep = "")
)

