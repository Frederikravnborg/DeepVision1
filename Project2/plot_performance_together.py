import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the 'Results' directory exists
os.makedirs('Results', exist_ok=True)

# Define the data
data_ph2_bce = {
    'EncDec': {
        'Training': {
            'Dice Coefficient': 0.9833,
            'IoU': 0.9150,
            'Accuracy': 0.9809,
            'Sensitivity': 0.9678,
            'Specificity': 0.9863
        },
        'Validation': {
            'Dice Coefficient': 0.9519,
            'IoU': 0.8016,
            'Accuracy': 0.9164,
            'Sensitivity': 0.8698,
            'Specificity': 0.9570
        },
        'Test': {
            'Dice Coefficient': 0.9649,
            'IoU': 0.8212,
            'Accuracy': 0.9482,
            'Sensitivity': 0.8974,
            'Specificity': 0.9687
        }
    },
    'UNet': {
        'Training': {
            'Dice Coefficient': 0.9813,
            'IoU': 0.9043,
            'Accuracy': 0.9791,
            'Sensitivity': 0.9619,
            'Specificity': 0.9860
        },
        'Validation': {
            'Dice Coefficient': 0.9734,
            'IoU': 0.8484,
            'Accuracy': 0.9680,
            'Sensitivity': 0.9463,
            'Specificity': 0.9791
        },
        'Test': {
            'Dice Coefficient': 0.9719,
            'IoU': 0.8537,
            'Accuracy': 0.9626,
            'Sensitivity': 0.9219,
            'Specificity': 0.9835
        }
    },
    'UNet2': {
        'Training': {
            'Dice Coefficient': 0.9826,
            'IoU': 0.9097,
            'Accuracy': 0.9779,
            'Sensitivity': 0.9716,
            'Specificity': 0.9788
        },
        'Validation': {
            'Dice Coefficient': 0.9602,
            'IoU': 0.8281,
            'Accuracy': 0.9364,
            'Sensitivity': 0.9467,
            'Specificity': 0.9261
        },
        'Test': {
            'Dice Coefficient': 0.9620,
            'IoU': 0.8263,
            'Accuracy': 0.9414,
            'Sensitivity': 0.9350,
            'Specificity': 0.9415
        }
    },
    'DilatedNet': {
        'Training': {
            'Dice Coefficient': 0.9789,
            'IoU': 0.8936,
            'Accuracy': 0.9766,
            'Sensitivity': 0.9581,
            'Specificity': 0.9845
        },
        'Validation': {
            'Dice Coefficient': 0.9635,
            'IoU': 0.8059,
            'Accuracy': 0.9482,
            'Sensitivity': 0.8990,
            'Specificity': 0.9579
        },
        'Test': {
            'Dice Coefficient': 0.9640,
            'IoU': 0.8129,
            'Accuracy': 0.9533,
            'Sensitivity': 0.9324,
            'Specificity': 0.9618
        }
    }
}

data_drive_bce = {
    'EncDec': {
        'Training': {
            'Dice Coefficient': 0.8864,
            'IoU': 0.0605,
            'Accuracy': 0.8721,
            'Sensitivity': 0.0000,
            'Specificity': 1.0000
        },
        'Validation': {
            'Dice Coefficient': 0.8902,
            'IoU': 0.0599,
            'Accuracy': 0.8817,
            'Sensitivity': 0.0000,
            'Specificity': 1.0000
        },
        'Test': {
            'Dice Coefficient': 0.8817,
            'IoU': 0.0614,
            'Accuracy': 0.8550,
            'Sensitivity': 0.0000,
            'Specificity': 1.0000
        }
    },
    'UNet': {
        'Training': {
            'Dice Coefficient': 0.9299,
            'IoU': 0.3199,
            'Accuracy': 0.8989,
            'Sensitivity': 0.6146,
            'Specificity': 0.9969
        },
        'Validation': {
            'Dice Coefficient': 0.9343,
            'IoU': 0.3304,
            'Accuracy': 0.8884,
            'Sensitivity': 0.5366,
            'Specificity': 0.9993
        },
        'Test': {
            'Dice Coefficient': 0.9195,
            'IoU': 0.3106,
            'Accuracy': 0.8816,
            'Sensitivity': 0.6185,
            'Specificity': 0.9819
        }
    },
    'UNet2': {
        'Training': {
            'Dice Coefficient': 0.9281,
            'IoU': 0.3776,
            'Accuracy': 0.9004,
            'Sensitivity': 0.8110,
            'Specificity': 0.9914
        },
        'Validation': {
            'Dice Coefficient': 0.9281,
            'IoU': 0.3466,
            'Accuracy': 0.9094,
            'Sensitivity': 0.8249,
            'Specificity': 0.9873
        },
        'Test': {
            'Dice Coefficient': 0.9311,
            'IoU': 0.3671,
            'Accuracy': 0.9058,
            'Sensitivity': 0.7654,
            'Specificity': 0.9937
        }
    },
    'DilatedNet': {
        'Training': {
            'Dice Coefficient': 0.8789,
            'IoU': 0.1613,
            'Accuracy': 0.8789,
            'Sensitivity': 0.3031,
            'Specificity': 0.9909
        },
        'Validation': {
            'Dice Coefficient': 0.8767,
            'IoU': 0.1374,
            'Accuracy': 0.8687,
            'Sensitivity': 0.0843,
            'Specificity': 0.9978
        },
        'Test': {
            'Dice Coefficient': 0.8781,
            'IoU': 0.1586,
            'Accuracy': 0.8736,
            'Sensitivity': 0.2040,
            'Specificity': 0.9974
        }
    }
}

data_loss_metrics = {
    'BCE': {
        'Training': {
            'Dice Coefficient': 0.9813,
            'IoU': 0.9043,
            'Accuracy': 0.9791,
            'Sensitivity': 0.9619,
            'Specificity': 0.9860
        },
        'Validation': {
            'Dice Coefficient': 0.9734,
            'IoU': 0.8484,
            'Accuracy': 0.9680,
            'Sensitivity': 0.9463,
            'Specificity': 0.9791
        },
        'Test': {
            'Dice Coefficient': 0.9719,
            'IoU': 0.8537,
            'Accuracy': 0.9626,
            'Sensitivity': 0.9219,
            'Specificity': 0.9835
        }
    },
    'BCE weight=2': {
        'Training': {
            'Dice Coefficient': 0.9568,
            'IoU': 0.8056,
            'Accuracy': 0.9615,
            'Sensitivity': 0.9724,
            'Specificity': 0.9548
        },
        'Validation': {
            'Dice Coefficient': 0.9548,
            'IoU': 0.7709,
            'Accuracy': 0.9493,
            'Sensitivity': 0.9135,
            'Specificity': 0.9621
        },
        'Test': {
            'Dice Coefficient': 0.9590,
            'IoU': 0.7984,
            'Accuracy': 0.9620,
            'Sensitivity': 0.9479,
            'Specificity': 0.9685
        }
    },
    'Focal': {
        'Training': {
            'Dice Coefficient': 0.7544,
            'IoU': 0.3238,
            'Accuracy': 0.3405,
            'Sensitivity': 1.0000,
            'Specificity': 0.0231
        },
        'Validation': {
            'Dice Coefficient': 0.7702,
            'IoU': 0.3661,
            'Accuracy': 0.3865,
            'Sensitivity': 1.0000,
            'Specificity': 0.0279
        },
        'Test': {
            'Dice Coefficient': 0.7538,
            'IoU': 0.3181,
            'Accuracy': 0.3202,
            'Sensitivity': 1.0000,
            'Specificity': 0.0250
        }
    }
}

# List of data dictionaries with their names and corresponding models
data_list = [
    ('PH2 BCE', data_ph2_bce, ['EncDec', 'UNet', 'UNet2', 'DilatedNet']),
    ('Drive BCE', data_drive_bce, ['EncDec', 'UNet', 'UNet2', 'DilatedNet']),
    ('Loss Metrics', data_loss_metrics, ['BCE', 'BCE weight=2', 'Focal'])
]

# Define metrics and datasets
metrics = ['Dice Coefficient', 'IoU', 'Accuracy', 'Sensitivity', 'Specificity']
datasets = ['Training', 'Validation', 'Test']
colors = {'Training': 'skyblue', 'Validation': 'lightgreen', 'Test': 'lightcoral'}
dataset_labels = list(colors.keys())

# Set up the figure and axes
fig, axs = plt.subplots(len(data_list), len(metrics), figsize=(25, 10), sharey=False)

# Iterate over each data dictionary (row)
for row_idx, (data_name, data_dict, models) in enumerate(data_list):
    # Iterate over each metric (column)
    for col_idx, metric in enumerate(metrics):
        ax = axs[row_idx, col_idx]
        x = np.arange(len(models))  # the label locations
        width = 0.2  # the width of the bars

        # Plot bars for each dataset
        for i, dataset in enumerate(datasets):
            metric_values = [data_dict[model][dataset][metric] for model in models]
            offset = (i - 1) * width  # Center the bars
            # Only add label to the first subplot to avoid duplicate legends
            label = dataset if (row_idx == 0 and col_idx == 0) else ""
            ax.bar(x + offset, metric_values, width, label=label, color=colors[dataset])

        # Add metric titles (column titles) with increased font size
        if row_idx == 0:
            ax.set_title(metric, fontsize=24, fontweight='bold')  # Increased font size

        # Add data dictionary names as y-axis labels (row titles) with substantially increased font size
        if col_idx == 0:
            ax.set_ylabel(data_name, fontsize=24, fontweight='bold')  # Substantially increased font size

        # Set x-axis labels (model names)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0, ha='center', fontsize=12)  # Increased font size for better readability

        # Set y-axis limits based on data
        ax.set_ylim(0, 1.05)  # Assuming all metrics are between 0 and 1

        # Add grid for better readability
        ax.grid(True, linestyle='--', axis='y', alpha=0.7)

# Add a single common legend
handles = [plt.Rectangle((0,0),1,1, color=colors[dataset]) for dataset in datasets]
fig.legend(handles, dataset_labels, loc='upper center', ncol=len(colors), fontsize=14)

# Adjust layout to prevent overlap and ensure adequate spacing
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust top spacing for the legend

# Save the figure
plt.savefig('Results/plot_combined.png', dpi=400)

# Show the plot
# plt.show()
