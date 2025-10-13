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



