"""
Generate README images for TFDV Lab1 (Titanic Dataset).
Creates visualizations that mirror the TFDV notebook workflow.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os
from sklearn.model_selection import train_test_split

IMG_DIR = 'img'
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

df = pd.read_csv('data/titanic.csv')
train_df, eval_df = train_test_split(df, test_size=0.2, shuffle=False)


# ── 1. TFDV Workflow Diagram ──────────────────────────────────────────
def create_workflow_diagram():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    boxes = [
        (1, 2.5, 'Training\nData', '#4285F4'),
        (4, 2.5, 'Compute\nStatistics', '#34A853'),
        (7, 2.5, 'Infer\nSchema', '#FBBC04'),
        (10, 2.5, 'Validate\nNew Data', '#EA4335'),
        (13, 2.5, 'Fix\nAnomalies', '#9C27B0'),
    ]

    for x, y, text, color in boxes:
        box = mpatches.FancyBboxPatch((x - 1.1, y - 0.8), 2.2, 1.6,
                                       boxstyle="round,pad=0.15",
                                       facecolor=color, edgecolor='white',
                                       linewidth=2, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')

    for i in range(len(boxes) - 1):
        ax.annotate('', xy=(boxes[i + 1][0] - 1.1, boxes[i + 1][1]),
                     xytext=(boxes[i][0] + 1.1, boxes[i][1]),
                     arrowprops=dict(arrowstyle='->', color='#555555', lw=2.5))

    ax.set_title('TFDV Workflow', fontsize=16, fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig(f'{IMG_DIR}/tfdv_workflow.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Created: tfdv_workflow.png')


# ── 2. Data Preview ──────────────────────────────────────────────────
def create_data_preview():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    preview = train_df.head(8).copy()
    preview['Name'] = preview['Name'].str[:25] + '...'
    preview['Ticket'] = preview['Ticket'].str[:10]

    table = ax.table(
        cellText=preview.values,
        colLabels=preview.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)

    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#4285F4')
            cell.set_text_props(color='white', fontweight='bold')
        elif key[0] % 2 == 0:
            cell.set_facecolor('#f0f4ff')
        cell.set_edgecolor('#dddddd')

    ax.set_title('Training Dataset Preview (First 8 Rows)', fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig(f'{IMG_DIR}/data_preview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Created: data_preview.png')


# ── 3. Descriptive Statistics ─────────────────────────────────────────
def create_statistics_overview():
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

    num_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass']

    for i, col in enumerate(num_cols):
        row, col_idx = divmod(i, 3)
        ax = fig.add_subplot(gs[row, col_idx])
        data = train_df[col].dropna()
        ax.hist(data, bins=30, color='#4285F4', edgecolor='white', alpha=0.8)
        ax.set_title(col, fontweight='bold')
        ax.set_ylabel('Count')
        stats_text = f'μ={data.mean():.1f}  σ={data.std():.1f}\nmin={data.min():.0f}  max={data.max():.0f}'
        ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                va='top', ha='right', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))

    ax_cat = fig.add_subplot(gs[1, 2])
    cat_features = {
        'Sex': train_df['Sex'].nunique(),
        'Embarked': train_df['Embarked'].nunique(),
        'Survived': train_df['Survived'].nunique(),
        'Pclass': train_df['Pclass'].nunique(),
    }
    missing = {
        'Age': train_df['Age'].isna().sum(),
        'Cabin': train_df['Cabin'].isna().sum(),
        'Embarked': train_df['Embarked'].isna().sum(),
    }
    text_lines = ['Categorical Features:']
    for k, v in cat_features.items():
        text_lines.append(f'  {k}: {v} unique values')
    text_lines.append('')
    text_lines.append('Missing Values:')
    for k, v in missing.items():
        pct = v / len(train_df) * 100
        text_lines.append(f'  {k}: {v} ({pct:.1f}%)')
    text_lines.append(f'\nTotal records: {len(train_df)}')

    ax_cat.axis('off')
    ax_cat.text(0.05, 0.95, '\n'.join(text_lines), transform=ax_cat.transAxes,
                fontsize=10, va='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#e8f0fe', edgecolor='#4285F4'))

    fig.suptitle('Training Dataset Statistics', fontsize=16, fontweight='bold', y=1.02)
    fig.savefig(f'{IMG_DIR}/train_statistics.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Created: train_statistics.png')


# ── 4. Schema Table ──────────────────────────────────────────────────
def create_schema_table():
    schema_data = {
        'Feature': ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
                     'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
        'Type': ['INT', 'INT', 'INT', 'STRING', 'STRING', 'FLOAT',
                 'INT', 'INT', 'STRING', 'FLOAT', 'STRING', 'STRING'],
        'Presence': ['required', 'required', 'required', 'required', 'required', 'required',
                     'required', 'required', 'required', 'required', 'required', 'required'],
        'Domain': ['-', '-', '-', '-', "'Sex'", 'min: 0; max: 100',
                   '-', '-', '-', 'min: 0; max: 600', '-', "'Embarked'"],
    }
    schema_df = pd.DataFrame(schema_data)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    table = ax.table(
        cellText=schema_df.values,
        colLabels=schema_df.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#34A853')
            cell.set_text_props(color='white', fontweight='bold')
        elif key[0] % 2 == 0:
            cell.set_facecolor('#f0fff0')
        cell.set_edgecolor('#dddddd')

    ax.set_title('Inferred Schema (from Training Data)', fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig(f'{IMG_DIR}/schema_table.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Created: schema_table.png')


# ── 5. Train vs Eval Statistics Comparison ────────────────────────────
def create_train_eval_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, col, color_t, color_e in zip(
        axes.flat,
        ['Age', 'Fare', 'Pclass', 'SibSp'],
        ['#4285F4', '#4285F4', '#4285F4', '#4285F4'],
        ['#EA4335', '#EA4335', '#EA4335', '#EA4335']
    ):
        train_data = train_df[col].dropna()
        eval_data = eval_df[col].dropna()
        bins = np.histogram_bin_edges(pd.concat([train_data, eval_data]), bins=25)
        ax.hist(train_data, bins=bins, alpha=0.6, color=color_t, label=f'Train (n={len(train_data)})', edgecolor='white')
        ax.hist(eval_data, bins=bins, alpha=0.6, color=color_e, label=f'Eval (n={len(eval_data)})', edgecolor='white')
        ax.set_title(col, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_ylabel('Count')

    fig.suptitle('Training vs Evaluation Dataset Statistics', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{IMG_DIR}/train_eval_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Created: train_eval_comparison.png')


# ── 6. Anomalies Table ───────────────────────────────────────────────
def create_anomalies_table():
    from util import add_extra_rows
    eval_with_anomalies = add_extra_rows(eval_df.copy())

    anomaly_data = {
        'Feature': ['Sex', 'Sex', 'Embarked', 'Age', 'Age', 'Fare'],
        'Anomaly': [
            'Unexpected string values',
            'Unexpected string values',
            'Unexpected string values',
            'Out of range (low)',
            'Out of range (high)',
            'Out of range (low)',
        ],
        'Description': [
            "Value 'unknown' not in schema domain",
            "Value 'non-binary' not in schema domain",
            "Value 'X' not in schema domain",
            "Min value -5.0 below expected range [0, 100]",
            "Max value 200.0 above expected range [0, 100]",
            "Min value -20.0 below expected range [0, 600]",
        ],
    }
    anom_df = pd.DataFrame(anomaly_data)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')

    table = ax.table(
        cellText=anom_df.values,
        colLabels=anom_df.columns,
        cellLoc='left',
        loc='center',
        colWidths=[0.12, 0.25, 0.55],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#EA4335')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('#fff0f0')
        cell.set_edgecolor('#dddddd')

    ax.set_title('Detected Anomalies in Evaluation Dataset', fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig(f'{IMG_DIR}/anomalies_detected.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Created: anomalies_detected.png')


# ── 7. Slice Comparison (Male vs Female) ─────────────────────────────
def create_slice_comparison():
    male_df = train_df[train_df['Sex'] == 'male']
    female_df = train_df[train_df['Sex'] == 'female']

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Survival rate
    ax = axes[0, 0]
    survival = pd.DataFrame({
        'Male': [male_df['Survived'].mean(), 1 - male_df['Survived'].mean()],
        'Female': [female_df['Survived'].mean(), 1 - female_df['Survived'].mean()]
    }, index=['Survived', 'Did not survive'])
    survival.plot(kind='bar', ax=ax, color=['#4285F4', '#EA4335'], edgecolor='white')
    ax.set_title('Survival Rate', fontweight='bold')
    ax.set_ylabel('Proportion')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(fontsize=9)

    # Age distribution
    ax = axes[0, 1]
    bins = np.linspace(0, 80, 25)
    ax.hist(male_df['Age'].dropna(), bins=bins, alpha=0.6, color='#4285F4', label='Male', edgecolor='white')
    ax.hist(female_df['Age'].dropna(), bins=bins, alpha=0.6, color='#EA4335', label='Female', edgecolor='white')
    ax.set_title('Age Distribution', fontweight='bold')
    ax.legend(fontsize=9)

    # Fare distribution
    ax = axes[0, 2]
    bins = np.linspace(0, 300, 30)
    ax.hist(male_df['Fare'].dropna(), bins=bins, alpha=0.6, color='#4285F4', label='Male', edgecolor='white')
    ax.hist(female_df['Fare'].dropna(), bins=bins, alpha=0.6, color='#EA4335', label='Female', edgecolor='white')
    ax.set_title('Fare Distribution', fontweight='bold')
    ax.legend(fontsize=9)

    # Pclass distribution
    ax = axes[1, 0]
    pclass_data = pd.DataFrame({
        'Male': male_df['Pclass'].value_counts().sort_index(),
        'Female': female_df['Pclass'].value_counts().sort_index()
    })
    pclass_data.plot(kind='bar', ax=ax, color=['#4285F4', '#EA4335'], edgecolor='white')
    ax.set_title('Passenger Class', fontweight='bold')
    ax.set_xlabel('Pclass')
    ax.set_xticklabels(['1st', '2nd', '3rd'], rotation=0)
    ax.legend(fontsize=9)

    # Embarked distribution
    ax = axes[1, 1]
    embarked_data = pd.DataFrame({
        'Male': male_df['Embarked'].value_counts().sort_index(),
        'Female': female_df['Embarked'].value_counts().sort_index()
    })
    embarked_data.plot(kind='bar', ax=ax, color=['#4285F4', '#EA4335'], edgecolor='white')
    ax.set_title('Embarkation Port', fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(fontsize=9)

    # Summary stats table
    ax = axes[1, 2]
    ax.axis('off')
    summary = [
        ['Metric', 'Male', 'Female'],
        ['Count', str(len(male_df)), str(len(female_df))],
        ['Survival Rate', f'{male_df["Survived"].mean():.1%}', f'{female_df["Survived"].mean():.1%}'],
        ['Mean Age', f'{male_df["Age"].mean():.1f}', f'{female_df["Age"].mean():.1f}'],
        ['Mean Fare', f'{male_df["Fare"].mean():.1f}', f'{female_df["Fare"].mean():.1f}'],
        ['Missing Age', f'{male_df["Age"].isna().sum()}', f'{female_df["Age"].isna().sum()}'],
    ]
    table = ax.table(cellText=summary[1:], colLabels=summary[0], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#9C27B0')
            cell.set_text_props(color='white', fontweight='bold')
        elif key[1] == 1:
            cell.set_facecolor('#e8eaf6')
        elif key[1] == 2:
            cell.set_facecolor('#fce4ec')
        cell.set_edgecolor('#dddddd')

    fig.suptitle('Slice Analysis: Male vs Female Passengers', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{IMG_DIR}/slice_male_vs_female.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Created: slice_male_vs_female.png')


# ── 8. Multi-slice: Sex × Embarked ───────────────────────────────────
def create_multi_slice():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, port, port_name in zip(axes, ['S', 'C', 'Q'], ['Southampton', 'Cherbourg', 'Queenstown']):
        port_df = train_df[train_df['Embarked'] == port]
        male_surv = port_df[port_df['Sex'] == 'male']['Survived'].mean()
        female_surv = port_df[port_df['Sex'] == 'female']['Survived'].mean()
        male_count = len(port_df[port_df['Sex'] == 'male'])
        female_count = len(port_df[port_df['Sex'] == 'female'])

        bars = ax.bar(['Male', 'Female'], [male_surv, female_surv],
                       color=['#4285F4', '#EA4335'], edgecolor='white', width=0.5)
        ax.set_title(f'{port_name} ({port})\nn={len(port_df)}', fontweight='bold')
        ax.set_ylabel('Survival Rate')
        ax.set_ylim(0, 1.05)

        for bar, count in zip(bars, [male_count, female_count]):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                    f'{bar.get_height():.1%}\n(n={count})', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Survival Rate by Sex × Embarkation Port', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{IMG_DIR}/slice_sex_embarked.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Created: slice_sex_embarked.png')


if __name__ == '__main__':
    create_workflow_diagram()
    create_data_preview()
    create_statistics_overview()
    create_schema_table()
    create_train_eval_comparison()
    create_anomalies_table()
    create_slice_comparison()
    create_multi_slice()
    print('\nAll images generated in img/ directory!')
