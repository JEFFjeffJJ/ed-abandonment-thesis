script_argument <- grep("^--file=", commandArgs(FALSE), value = TRUE)
project_dir <- dirname(dirname(normalizePath(sub("^--file=", "", script_argument))))
data_dir <- file.path(project_dir, "data", "processed")
table_dir <- file.path(project_dir, "outputs", "tables")

estimate_scenario <- function(scenario, background, bundle) {
  model <- bundle$model
  data <- background
  data$previous_ed_visits_log <- log1p(scenario$previous_ed_visits)
  data$triage_acuity_factor <- factor(as.character(scenario$triage_acuity),
                                     levels = model$xlevels$triage_acuity_factor)
  data$acuity_missing <- 0
  if ("arrival_transport_factor" %in% names(model$xlevels)) {
    data$arrival_transport_factor <- factor(scenario$arrival_transport,
                                    levels = model$xlevels$arrival_transport_factor)
  }
  x <- model.matrix(delete.response(terms(model)), data,
                    contrasts.arg = model$contrasts, xlev = model$xlevels)
  p <- plogis(as.vector(x %*% coef(model)))
  weights <- data$weight / sum(data$weight)
  average <- sum(p * weights)
  gradient <- colSums(x * p * (1 - p) * weights)
  se <- sqrt(as.numeric(t(gradient) %*% bundle$cluster_vcov %*% gradient))
  # Logit-scale delta intervals stay inside [0, 1].
  logit_se <- se / (average * (1 - average))
  data.frame(adjusted_probability = average,
             confidence_interval_lower = plogis(qlogis(average) - 1.96 * logit_se),
             confidence_interval_upper = plogis(qlogis(average) + 1.96 * logit_se))
}

main <- function() {
  raw <- read.csv(gzfile(file.path(data_dir, "regression_data.csv.gz")),
                  stringsAsFactors = FALSE)
  bundles <- readRDS(file.path(data_dir, "logistic_regression_model.rds"))
  results <- list()
  for (cohort in c("A_full", "C_transport")) {
    data <- raw
    if (cohort == "C_transport") data <- data[!is.na(data$arrival_transport) &
                                               data$arrival_transport != "", ]
    data$triage_pain[data$pain_missing == 1] <- bundles[[cohort]]$pain_median
    data$weight <- 1
    background <- aggregate(weight ~ triage_pain + pain_missing, data, sum)
    grid <- expand.grid(previous_ed_visits = c(0, 1, 3, 5, 10), triage_acuity = 1:5,
      arrival_transport = if (cohort == "A_full") "ALL" else c("WALK IN", "AMBULANCE"),
      stringsAsFactors = FALSE)
    rows <- lapply(seq_len(nrow(grid)), function(i) {
      scenario <- grid[i, ]
      support <- data[data$acuity_missing == 0 &
        data$triage_acuity == scenario$triage_acuity &
        data$previous_ed_visits == scenario$previous_ed_visits, ]
      if (cohort == "C_transport")
        support <- support[support$arrival_transport == scenario$arrival_transport, ]
      n <- nrow(support)
      events <- sum(support$abandonment)
      cbind(cohort = cohort, scenario, support_visits = n, support_events = events,
            display_supported = n >= 100 & events >= 5,
            estimate_scenario(scenario, background, bundles[[cohort]]))
    })
    results[[cohort]] <- do.call(rbind, rows)
  }
  output <- do.call(rbind, results)
  for (column in c("adjusted_probability", "confidence_interval_lower",
                   "confidence_interval_upper"))
    output[[paste0(column, "_percent")]] <- output[[column]] * 100
  write.csv(output, file.path(table_dir, "scenario_probabilities.csv"), row.names = FALSE)
  print(aggregate(display_supported ~ cohort, output, sum), row.names = FALSE)
  cat("Support counts and conditional model-based intervals saved; unsupported\n",
      "profiles remain in the audit table but are excluded from figures.\n")
}
if (sys.nframe() == 0L) main()
