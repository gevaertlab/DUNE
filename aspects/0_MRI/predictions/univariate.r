suppressMessages({
    library("tidyverse")
    library("rjson")
    library("umap")
    library("cowplot")
    library("optparse")
    library("caret")
    library("cowplot")
    library("ggthemes")
    library("survival")
    library("svMisc")
    library("crayon")
})

grouped_wilcox <- function(variable, df = df, num_feats) {
    num_feats <- ncol(features)
    variable <- as.factor(unlist(labels[, variable]))
    pvals <- numeric()
    for (i in 1:num_feats) {
        feature <- as.numeric(unlist(features[, i]))
        pval <- wilcox.test( feature ~ variable)$p.value
        pvals[i] <- pval
        progress(i, num_feats)
    }
    pvals = pvals * num_feats
    res = sum(pvals < 0.05, na.rm = T) / num_feats
    return(res)
}

grouped_logrank <- function(variable, features, labels) {
    num_feats = ncol(features)
    time <- paste(variable, "_delay", sep="")
    event <- paste(variable, "_event", sep = "")
    
    time <- as.numeric(unlist(labels[, time]))
    event <- as.integer(unlist(labels[, event]))

    pvals <- numeric()
    for (i in 1:num_feats) {
        feature <- as.numeric(unlist(features[, i]))

        d = data.frame(time = time, event = event, feature = feature)

        cox = coxph(Surv(time, event) ~ feature, data = d)
        pvals[i] = summary(cox)$coefficients[5]
        progress(i, num_feats)
    }
    pvals = pvals * num_feats
    res = sum(pvals < 0.05, na.rm=T) / num_feats
    return(res)
}

grouped_anova <- function(variable, df, num_feats) {
    num_feats <- ncol(features)
    variable <- as.factor(unlist(labels[, variable]))
    pvals <- numeric()

    for (i in 1:num_feats) {
        feature <- as.numeric(unlist(features[, i]))
        pval <- anova(lm(feature ~ variable))[["Pr(>F)"]][1]
        pvals[i] <- pval
        progress(i, num_feats)
    }
    pvals = pvals * num_feats
    res = sum(pvals < 0.05, na.rm = T) / num_feats
    return(res)
}

grouped_spearman <- function(variable, features, labels) {

    num_feats <- ncol(features)
    variable <- as.numeric(unlist(labels[, variable]))
    pvals <- numeric()
    for (i in 1:num_feats) {
        feature <- as.numeric(unlist(features[, i]))
        pval <- cor.test(x = feature, y = variable, method="spearman")$p.value
        pvals[i] = pval
        progress(i, num_feats)
    }
    pvals = pvals * num_feats
    res = sum(pvals < 0.05, na.rm = T) / num_feats
    return(res)
}

draw_umap <- function(umap, labels, var_name, var_class) {
    var = labels %>% pull(var_name)
    p <- data.frame(
        x = umap$layout[, 1],
        y = umap$layout[, 2],
        var = var  ) %>%
        ggplot(aes_string("x", "y", col = var)) +
            geom_point(size = 0.7) +
            theme_tufte() +
            theme(legend.position = "None") +
            ggtitle(var_name) 
    
    ggsave(paste(OUTPUT_PATH, "/",  var_name,".pdf", sep=""), p)
    return(p)
}

shannon_entropy <- function(cat.vect) {
    px <- table(cat.vect) / length(cat.vect)
    lpx <- log(px, base = 2)
    ent <- -sum(px * lpx)
    return(ent)
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
message("\nImporting data...")
METADATA <- config$metadata_path
VARIABLES <- config$variables_path
FEATURES <- paste(config$model_path, config$feature_path, sep = "")
OUTPUT_PATH <- paste(config$model_path, config$output_path, sep = "")
UMAP <- as.logical(config$umap)

TAF::mkdir(OUTPUT_PATH)


features = bind_rows(
    data.table::fread(paste(FEATURES, "train_features.csv.gz", sep=""), header=T),
    data.table::fread(paste(FEATURES, "test_features.csv.gz", sep=""), header=T)  
) %>%
    rename(eid = V1) %>%
    mutate(eid = substr(eid, 1, 7)) %>%
    as.data.frame()

num_feats = ncol(features)

metadata <- data.table::fread(METADATA, stringsAsFactors = T) %>%
    mutate(eid = as.character(eid)) %>% as.data.frame()    
variables <- data.table::fread(VARIABLES) %>% as.data.frame()

factors <- variables$var[variables$task == "classification"]
metadata <- metadata %>%
    mutate_at(factors, as.factor)
    

# FUSED DF
df <- metadata %>% right_join(features, by = ("eid")) %>% column_to_rownames("eid")
labels <- df %>% select(!num_range("", 0:num_feats))
features <- df %>% select(num_range("", 0:num_feats))



# ANALYSIS
if (UMAP) {
    message("\nUMAP analysis...")
    Umap <- umap(df %>% select(num_range("", 0:50000)))
}

message("\nComputing univariate analyses...")
results <- tibble(variables) %>% filter(keep_model)
results$proportion_sig = NA
results$variance = NA
num_var <- length(results$var)
plot_list <- list()
i <- 1
for (i in 1:num_var) {
    variable = results$var[i]
    task = results$task[i]
    res = NA
    variance = NA
    num_classes = NA
    message(yellow("\n", variable, " ", task, " ", i, "/", num_var, sep=""))

    if (task == "regression") {
        variance = var(labels[, variable])
        res = grouped_spearman(variable, features, labels)
    } else if (task == "classification") {
        num_classes = unique(labels[, variable])
        variance = shannon_entropy(labels[, variable])
        lvls <- length(levels(df %>% pull(variable)))
        if (lvls < 3) {
            res = grouped_wilcox(variable, features, labels)
        } else {
            res = grouped_anova(variable, features, labels)
        }
    } else {
        res = grouped_logrank(variable, features, labels)
    }

    print(res)
    results[results$var == variable, "proportion_sig"] = round(res, 4)
    results[results$var == variable, "variance"] = round(variance, 4)
    results[results$var == variable, "num_classes"] = num_classes

    write_csv(
        results,
        file = paste(OUTPUT_PATH, "/0-univariate_results.csv", sep = "")
    )

    if ( UMAP & task != "survival" ) {
        plot_list[[i]] = draw_umap(umap = Umap, labels = labels, var_name = variable, var_class = task)
    }
}

if (UMAP) {
    ncols = 10
    num_plots = length(plot_list)
    pl <- plot_grid(plotlist = plot_list, rel_heights = 1, rel_widths = 1, ncol = ncols) + theme(legend.position = "None")
    # ggsave(paste(OUTPUT_PATH, "/0-umaps.pdf", sep = ""), pl,
    #     height = 5.5 * num_plots %/% ncols,
    #     width = 5 * ncols,
    #     units = "cm"
    # )
    
    ggsave(paste(OUTPUT_PATH, "/0-umaps.jpg", sep = ""), pl,
        height = 7 * num_plots %/% ncols, 
        width = 5.5 * ncols, 
        units = "cm"
    )
}


