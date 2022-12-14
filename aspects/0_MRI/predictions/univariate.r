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

##################################
##################################

# age	regression
# sex	classification
# sleep	regression
# height	regression
# weight	regression
# bmi	regression
# diastole	regression
# smoking	classification
# alcohol_freq	regression
# alcohol_status	classification
# greymat_vol	regression
# brain_vol	regression
# norm_brainvol	regression
# fluidency	regression
# digits_symbols	regression
# depression	classification



# DATA = "/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/UNet_5b_8f_UKfull/autoencoding/features/concat_train.csv"

DATA = paste(config$model_path, config$train_csv_path, sep = "")
config$output_path = paste(config$model_path, "/univariate", sep = "")
TAF::mkdir(config$output_path)
    
df <- data.table::fread(DATA) %>%
    as_tibble() %>%
    select(-case) %>%
    mutate(
        age = age,
        sex = as.factor(sex),
        sleep = sleep,
        height = height,
        weight = weight,
        bmi = bmi,
        diastole = diastole,
        smoking = as.factor(if_else(smoking == "Never", "No", "Yes")),
        alcohol_freq = as.numeric(ordered(alcohol_freq, c("Never", "Special occasions only", "One to three times a month", "Once or twice a week", "Three or four times a week", "Daily or almost daily"))),
        alcohol_status = as.factor(if_else(alcohol_status == "Never", "No", "Yes")),
        greymat_vol = greymat_vol,
        brain_vol = brain_vol,
        norm_brainvol = norm_brainvol,
        fluidency = fluidency,
        digits_symbols = digits_symbols,
        depression = as.factor(if_else(depression == "No", "No", "Yes"))
    )


results <- data.table::fread("data/metadata/UKB_variables.csv") %>% as_tibble()
results$res <- numeric(nrow(results))


umap <- umap(df %>% select(num_range("", 0:50000)))
labels <- df %>%
    select(!num_range("", 0:50000)) %>%
    mutate(across(where(~ all(unique(.[!is.na(.)]) %in% c("0", "1"))), as.factor))


for (var in colnames(labels)) {
    print(var)
    isFactor <- is.factor(labels %>% select(var) %>% pull())
    if (isFactor) {
        res = grouped_wilcox(variable = var, df = df)
    } else {
        res = grouped_spearman(variable = var, df = df)
    }
    results[results$var == var, "proportion_sig"] = res
    draw_umap(umap=umap, labels=labels, var = var)
}


write_csv(
    results,
    file = paste(paste(config$output_path, "/0-univariate_results.csv", sep = ""))
)

