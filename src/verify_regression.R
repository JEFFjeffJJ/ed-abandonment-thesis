script_argument <- grep("^--file=", commandArgs(FALSE), value = TRUE)
project_dir <- dirname(dirname(normalizePath(sub("^--file=", "", script_argument))))
source(file.path(project_dir, "src", "regression_analysis.R"))
source(file.path(project_dir, "src", "scenario_analysis.R"))

raw <- read.csv(gzfile(file.path(data_dir, "abandonment_modeling_data.csv.gz")),
                stringsAsFactors = FALSE)
bundles <- readRDS(file.path(data_dir, "logistic_regression_model.rds"))
checks <- list()
for (cohort in names(bundles)) {
  data <- raw
  if (cohort != "A_full") data <- data[!is.na(data$arrival_transport) &
                                        data$arrival_transport != "", ]
  bundle <- bundles[[cohort]]
  recalculated <- calculate_cluster_robust_results(bundle$model, data$subject_id)
  difference <- max(abs(recalculated$cluster_vcov - bundle$cluster_vcov))
  stopifnot(difference < 1e-10,
            min(eigen(bundle$cluster_vcov, symmetric = TRUE)$values) > -1e-10,
            abs(mean(fitted(bundle$model)) - mean(data$abandonment)) < 1e-8)
  checks[[cohort]] <- data.frame(cohort = cohort, visits = nrow(data),
                                 max_covariance_difference = difference,
                                 r_version = R.version.string)
}
scenarios <- read.csv(file.path(table_dir, "scenario_probabilities.csv"))
data <- prepare_regression_data(raw)
data$weight <- 1
background <- aggregate(weight ~ triage_pain + pain_missing, data, sum)
selected <- scenarios[scenarios$cohort == "A_full" & scenarios$triage_acuity == 3 &
                        scenarios$previous_ed_visits == 0, ]
recreated <- estimate_scenario(selected, background, bundles$A_full)
stopifnot(abs(recreated$adjusted_probability - selected$adjusted_probability) < 1e-12)
write.csv(do.call(rbind, checks), file.path(table_dir, "regression_verification.csv"),
          row.names = FALSE)
cat("Regression covariance, prevalence, and scenario roundtrip checks passed.\n")
