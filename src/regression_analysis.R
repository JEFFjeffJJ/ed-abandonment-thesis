script_argument <- grep("^--file=", commandArgs(FALSE), value = TRUE)
project_dir <- dirname(dirname(normalizePath(sub("^--file=", "", script_argument))))
data_dir <- file.path(project_dir, "data", "processed")
table_dir <- file.path(project_dir, "outputs", "tables")

prepare_regression_data <- function(data) {
  data$acuity_missing <- as.integer(is.na(data$triage_acuity))
  data$pain_missing <- as.integer(is.na(data$triage_pain))
  data$triage_acuity[is.na(data$triage_acuity)] <- median(data$triage_acuity, na.rm = TRUE)
  data$triage_pain[is.na(data$triage_pain)] <- median(data$triage_pain, na.rm = TRUE)
  data$previous_ed_visits_log <- log1p(data$previous_ed_visits)
  data$triage_acuity_factor <- relevel(factor(data$triage_acuity, levels = 1:5), ref = "3")
  data$arrival_transport_factor <- relevel(factor(data$arrival_transport), ref = "WALK IN")
  data$history_group <- cut(data$previous_ed_visits, c(-1, 0, 1, 3, 5, 10, Inf),
                           labels = c("0", "1", "2-3", "4-5", "6-10", ">10"))
  data
}

fit_logistic_regression <- function(data, outcome = "abandonment", transport = FALSE,
                                    complete_case = FALSE, categorical_history = FALSE) {
  predictors <- c(if (categorical_history) "history_group" else "previous_ed_visits_log",
                  "triage_acuity_factor", "triage_pain")
  if (transport) predictors <- c(predictors, "arrival_transport_factor")
  if (!complete_case) predictors <- c(predictors, "acuity_missing", "pain_missing")
  glm(reformulate(predictors, outcome), data = data, family = binomial(),
      control = glm.control(epsilon = 1e-10, maxit = 100))
}

calculate_cluster_robust_results <- function(model, patient_ids) {
  x <- model.matrix(model)
  stopifnot(nrow(x) == length(patient_ids), model$converged, !anyNA(coef(model)))
  # Patients, not visits, are the independent units of the sandwich estimator.
  scores <- rowsum(x * residuals(model, type = "response"), patient_ids, reorder = FALSE)
  bread <- solve(crossprod(x, x * model$weights))
  g <- nrow(scores)
  n <- nrow(x)
  k <- ncol(x)
  vcov <- (g / (g - 1)) * ((n - 1) / (n - k)) *
    bread %*% crossprod(scores) %*% bread
  beta <- coef(model)
  se <- sqrt(diag(vcov))
  results <- data.frame(term = names(beta), coefficient = unname(beta),
    robust_standard_error = unname(se), odds_ratio = exp(unname(beta)),
    confidence_interval_lower = exp(unname(beta - 1.96 * se)),
    confidence_interval_upper = exp(unname(beta + 1.96 * se)),
    p_value = unname(2 * pnorm(abs(beta / se), lower.tail = FALSE)))
  list(results = results, cluster_vcov = vcov)
}

calculate_joint_wald_test <- function(model, vcov, pattern) {
  terms <- grep(pattern, names(coef(model)), value = TRUE)
  b <- coef(model)[terms]
  statistic <- as.numeric(t(b) %*% solve(vcov[terms, terms, drop = FALSE]) %*% b)
  data.frame(degrees_of_freedom = length(terms), wald_chi_squared = statistic,
             p_value = pchisq(statistic, length(terms), lower.tail = FALSE))
}

test_interactions <- function(raw) {
  # The former helicopter interaction had too few outcomes and separation.
  data <- raw[raw$arrival_transport %in% c("WALK IN", "AMBULANCE") &
                !is.na(raw$triage_acuity), ]
  data <- prepare_regression_data(data)
  formula <- abandonment ~ previous_ed_visits_log * triage_acuity_factor +
    previous_ed_visits_log * arrival_transport_factor + triage_pain + pain_missing
  model <- glm(formula, data, family = binomial(),
               control = glm.control(epsilon = 1e-10, maxit = 100))
  tighter <- glm(formula, data, family = binomial(),
                 control = glm.control(epsilon = 1e-12, maxit = 100))
  difference <- max(abs(coef(model) - coef(tighter)))
  stopifnot(model$converged, tighter$converged, difference < 0.01)
  robust <- calculate_cluster_robust_results(model, data$subject_id)
  tests <- rbind(
    data.frame(interaction = "History x acuity",
      calculate_joint_wald_test(model, robust$cluster_vcov,
                               "^previous_ed_visits_log:triage_acuity_factor")),
    data.frame(interaction = "History x transport",
      calculate_joint_wald_test(model, robust$cluster_vcov,
                               "^previous_ed_visits_log:arrival_transport_factor")))
  tests$visits <- nrow(data)
  tests$patients <- length(unique(data$subject_id))
  tests$max_coefficient_change_tighter_tolerance <- difference
  tests$population <- "Walk-in/ambulance, observed acuity"
  write.csv(tests, file.path(table_dir, "interaction_test_results.csv"), row.names = FALSE)
  write.csv(robust$results, file.path(table_dir, "interaction_model_coefficients.csv"),
            row.names = FALSE)
  support <- aggregate(cbind(visits = rep(1, nrow(data)), events = data$abandonment),
    by = list(transport = data$arrival_transport, acuity = data$triage_acuity,
              returning = data$previous_ed_visits > 0), FUN = sum)
  write.csv(support, file.path(table_dir, "interaction_support.csv"), row.names = FALSE)
}

main <- function() {
  raw <- read.csv(gzfile(file.path(data_dir, "abandonment_modeling_data.csv.gz")),
                  stringsAsFactors = FALSE)
  known <- raw[!is.na(raw$arrival_transport) & raw$arrival_transport != "", ]
  datasets <- list(A_full = raw, B_known = known, C_transport = known)
  results <- list()
  summaries <- list()
  bundles <- list()
  for (name in names(datasets)) {
    data <- prepare_regression_data(datasets[[name]])
    model <- fit_logistic_regression(data, transport = name == "C_transport")
    robust <- calculate_cluster_robust_results(model, data$subject_id)
    results[[name]] <- cbind(analysis = name, outcome = "abandonment", robust$results)
    summaries[[name]] <- data.frame(analysis = name, visits = nrow(data),
      patients = length(unique(data$subject_id)), events = sum(data$abandonment),
      acuity_missing = sum(data$acuity_missing), pain_missing = sum(data$pain_missing))
    bundles[[name]] <- list(model = model, cluster_vcov = robust$cluster_vcov,
      acuity_median = median(datasets[[name]]$triage_acuity, na.rm = TRUE),
      pain_median = median(datasets[[name]]$triage_pain, na.rm = TRUE),
      r_version = R.version.string)
  }
  # Each component is contrasted with every other documented disposition.
  for (outcome in c("lwbs", "eloped", "ama")) {
    data <- prepare_regression_data(raw)
    model <- fit_logistic_regression(data, outcome = outcome)
    robust <- calculate_cluster_robust_results(model, data$subject_id)
    results[[outcome]] <- cbind(analysis = "A_component", outcome, robust$results)
  }
  complete <- raw[complete.cases(raw[c("triage_acuity", "triage_pain")]), ]
  data <- prepare_regression_data(complete)
  model <- fit_logistic_regression(data, complete_case = TRUE)
  results[["complete"]] <- cbind(analysis = "A_complete_case", outcome = "abandonment",
    calculate_cluster_robust_results(model, data$subject_id)$results)
  summaries[["complete"]] <- data.frame(analysis = "A_complete_case", visits = nrow(data),
    patients = length(unique(data$subject_id)), events = sum(data$abandonment),
    acuity_missing = 0, pain_missing = 0)
  data <- prepare_regression_data(raw)
  model <- fit_logistic_regression(data, categorical_history = TRUE)
  results[["categorical"]] <- cbind(analysis = "A_categorical_history", outcome = "abandonment",
    calculate_cluster_robust_results(model, data$subject_id)$results)
  write.csv(do.call(rbind, results), file.path(table_dir, "logistic_regression_results.csv"),
            row.names = FALSE)
  write.csv(do.call(rbind, summaries), file.path(table_dir, "regression_sample_summary.csv"),
            row.names = FALSE)
  saveRDS(bundles, file.path(data_dir, "logistic_regression_model.rds"))
  connection <- gzfile(file.path(data_dir, "regression_data.csv.gz"), "wt")
  write.csv(prepare_regression_data(raw), connection, row.names = FALSE)
  close(connection)
  test_interactions(raw)
  print(do.call(rbind, summaries), row.names = FALSE)
  cat("Regression, component and sensitivity analyses saved.\n")
}
if (sys.nframe() == 0L) main()
