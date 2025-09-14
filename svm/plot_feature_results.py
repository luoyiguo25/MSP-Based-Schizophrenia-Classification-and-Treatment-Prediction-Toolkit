from matplotlib import pyplot as plt
import pandas as pd

# Set parameters
t = 1  # Specific value for t
csv_filename = f'analysis_features_contributions_t{t}.csv'

# Load data from CSV file
results_df = pd.read_csv(csv_filename)

# --------------------------
# Plot line charts for 4 metrics
# --------------------------
# Extract metrics
feature_removed = results_df['Feature Removed'].values
auc = results_df['AUC'].values
accuracy = results_df['Accuracy'].values
specificity = results_df['Specificity'].values
sensitivity = results_df['Sensitivity'].values

# Create figure with subplots
plt.figure(figsize=(10, 6))

# Plot AUC
plt.subplot(2, 2, 1)
plt.plot(feature_removed, auc, marker='o')
plt.xlabel('Feature Removed')
plt.ylabel('AUC')
plt.axhline(y=auc[-1], color='gray', linestyle='--')  # Baseline reference
plt.xticks(rotation=45, ha='right')

# Plot Accuracy
plt.subplot(2, 2, 2)
plt.plot(feature_removed, accuracy, marker='o')
plt.xlabel('Feature Removed')
plt.ylabel('Accuracy')
plt.axhline(y=accuracy[-1], color='gray', linestyle='--')  # Baseline reference
plt.xticks(rotation=45, ha='right')

# Plot Specificity
plt.subplot(2, 2, 3)
plt.plot(feature_removed, specificity, marker='o')
plt.xlabel('Feature Removed')
plt.ylabel('Specificity')
plt.axhline(y=specificity[-1], color='gray', linestyle='--')  # Baseline reference
plt.xticks(rotation=45, ha='right')

# Plot Sensitivity
plt.subplot(2, 2, 4)
plt.plot(feature_removed, sensitivity, marker='o')
plt.xlabel('Feature Removed')
plt.ylabel('Sensitivity')
plt.axhline(y=sensitivity[-1], color='gray', linestyle='--')  # Baseline reference
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

# --------------------------
# Plot bar chart for AUC difference
# --------------------------
# Calculate relative AUC difference
baseline_auc = results_df.iloc[-1]['AUC']
plot_auc = (results_df['AUC'] - baseline_auc) / baseline_auc * 100
plot_auc = plot_auc.iloc[:-1]  # Remove last row

# Define x-axis labels
xlabel = ['SV_22q', 'SV_adhd', 'SV_asd', 'SV_bd', 'SV_mdd', 'SV_ep', 'SV_ocd', 'SV_scz', 
          'CT_22q', 'CT_adhd', 'CT_asd', 'CT_bd', 'CT_mdd', 'CT_ep', 'CT_ocd', 'CT_scz']

# Create bar chart
plt.figure(figsize=(15/2.54, 10/2.54))
plt.bar(xlabel, plot_auc, color='#5b81a6')

# Customize plot appearance
plt.ylabel('△AUC (%)', fontsize=10)
plt.xticks(rotation=45, fontsize=10, ha='right')
plt.yticks(fontsize=10)
plt.ylim([-1.8, 1])
plt.axhline(0, color='black', linewidth=1)  # Horizontal reference line

# Remove top and right spines
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.tight_layout()

# Save and show plot
plt.rcParams['svg.fonttype'] = 'none'  # Preserve text as editable
plt.savefig(f'pictures/auc_bar_{t}.svg', format='svg')
plt.show()
