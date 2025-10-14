library(tidyverse)
library(logisticPCA)
library(ggplot2)

background <- read_csv('data/background-clean.csv')

# Extract class features (cols 11–29)
classes_df <- background %>% select(11:29)
X <- as.matrix(classes_df)  # logisticPCA expects a matrix of 0/1

set.seed(100)
k_fit <- 4                            # use 3–6 typically; 4 is a common start
fit <- logisticPCA(X, k = k_fit, m = 1, main_effects = TRUE)

scores <- fit$PCs                     # n × k_fit
colnames(scores) <- paste0("PC", seq_len(ncol(scores)))

# Cluster on all k_fit components for better structure
km <- kmeans(scores, centers = 3, nstart = 50)

# Plot only the first two components (projection)
plot_df <- as.data.frame(scores[, 1:2, drop = TRUE]) %>%
  setNames(c("PC1","PC2")) %>%
  mutate(cluster = factor(km$cluster))

ggplot(plot_df, aes(PC1, PC2, color = cluster)) +
  geom_point(alpha = 0.8) +
  labs(title = sprintf("Logistic PCA (k=%d) — clusters shown on PC1–PC2", k_fit),
       color = "Cluster") +
  theme_minimal()

# Optional profiles in original class space
cluster_profiles <- as.data.frame(X) %>%
  mutate(cluster = km$cluster) %>%
  group_by(cluster) %>%
  summarise(across(everything(), mean), .groups = "drop")
cluster_profiles


library(dplyr)
library(tidyr)
library(stringr)

background_with_cluster_ids <- background %>%
  mutate(cluster = km$cluster) %>%
  group_by(cluster) %>%
  mutate(cluster_size = n()) %>%   # size per cluster on each row
  ungroup()

prof_vars <- c("prog.prof","math.prof","stat.prof")
comf_vars <- c("prog.comf","math.comf","stat.comf")

# ---- Proficiency means (1=beg, 2=int, 3=adv) + size ----
prof_means <- background_with_cluster_ids %>%
  mutate(across(all_of(prof_vars),
                ~ as.integer(factor(.x, levels = c("beg","int","adv"), ordered = TRUE)))) %>%
  group_by(cluster) %>%
  summarise(
    cluster_size = n(),
    across(all_of(prof_vars), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )

prof_means

# ---- All means (proficiency + comfort) + size ----
cluster_means_all <- background_with_cluster_ids %>%
  mutate(across(all_of(prof_vars),
                ~ as.integer(factor(.x, levels = c("beg","int","adv"), ordered = TRUE)))) %>%
  group_by(cluster) %>%
  summarise(
    cluster_size = n(),
    across(c(all_of(prof_vars), all_of(comf_vars)), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  relocate(cluster_size, .after = cluster)

cluster_means_all

library(knitr)

cluster_means_all %>%
  kable(
    digits = 2,
    col.names = c("Cluster",
                  "Cluster Size",
                  "prog.prof",
                  "math.prof",
                  "stat.prof",
                  "prog.comf",
                  "math.comf",
                  "stat.comf"),
    caption = "Cluster-wise Means of Proficiency and Comfort (with Cluster Sizes)"
  )

vars <- c("prog.prof","math.prof","stat.prof","prog.comf","math.comf","stat.comf")

####################################################################################
# Create a graph with a line for each cluster using Z scores to show highs and lows
####################################################################################

line_df <- cluster_means_all %>%
  select(cluster, cluster_size, all_of(vars)) %>%
  pivot_longer(cols = all_of(vars), names_to = "variable", values_to = "mean") %>%
  group_by(variable) %>%
  mutate(z = (mean - mean(mean)) / sd(mean)) %>%
  ungroup() %>%
  mutate(variable = factor(variable, levels = vars))

ggplot(line_df, aes(x = variable, y = z,
                    group = factor(cluster), color = factor(cluster))) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  labs(x = NULL, y = "z-score (within variable)",
       color = "Cluster",
       title = "Cluster profiles across proficiency & comfort (z-scores)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_color_discrete(
    labels = function(lv) {
      sizes <- cluster_means_all$cluster_size[match(as.integer(lv),
                                                    cluster_means_all$cluster)]
      paste0("C", lv, " (n=", sizes, ")")
    }
  )

####################################################################################
# Using means from clusters calculated in week2-surveys.R, create a graph with same
# format as the previous graph.
####################################################################################

c1_means <- c( 2.278, 2.500, 2.722, 3.500, 3.611, 4.000)
c2_means <- c( 2.619, 2.714, 2.952, 4.286, 4.714, 4.667)
c3_means <- c( 2.000, 1.750, 1.750, 3.667, 3.500, 3.000)

# Cluster sizes for the new analysis
sizes_B <- c(C1 = 18, C2 = 21, C3 = 12)

# ---- Build the summary table for the new analysis (call it Solution B) ----
means_B <- rbind(c1_means, c2_means, c3_means) %>%
  as.data.frame()
colnames(means_B) <- vars
means_B <- means_B %>%
  mutate(cluster = 1:3,
         cluster_size = c(sizes_B["C1"], sizes_B["C2"], sizes_B["C3"])) %>%
  relocate(cluster, cluster_size)

# ---- Make the z-score line plot (three lines, one per cluster) ----
line_df_B <- means_B %>%
  select(cluster, cluster_size, all_of(vars)) %>%
  pivot_longer(cols = all_of(vars), names_to = "variable", values_to = "mean") %>%
  group_by(variable) %>%
  mutate(z = (mean - mean(mean)) / sd(mean)) %>%  # standardize within variable
  ungroup() %>%
  mutate(variable = factor(variable, levels = vars))

ggplot(line_df_B, aes(x = variable, y = z,
                      group = factor(cluster), color = factor(cluster))) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  labs(x = NULL, y = "z-score (within variable)",
       color = "Cluster",
       title = "Solution B — Cluster profiles across proficiency & comfort (z-scores)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_color_discrete(
    labels = function(lv) {
      sizes <- means_B$cluster_size[match(as.integer(lv), means_B$cluster)]
      paste0("C", lv, " (n=", sizes, ")")
    }
  )
