import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf


def run_and_show_regression(y, xs, df, datasetname="hf"):
    # Define the formula for the regression
    formula = f"{y} ~ {' + '.join(xs)}"  # + "+ company1 : company2"

    # Create the regression model
    model = smf.ols(formula=formula, data=df)

    # Fit the model to the data
    results = model.fit()

    # Print the summary of the regression results
    print(results.summary())

    latex = results.summary().as_latex()
    with open(f"../final_figures/{y}_{datasetname}_regression.tex", "w") as f:
        f.write(latex)


def plot_heatmap(
    df=None,
    col="",
    dfcorr=None,
    order_with_clustermap=True,
    fontsize=15,
    save=False,
    savename="",
    do_mask=False,
):
    if dfcorr is None:
        dfcopy = df.copy()[["model_1", "model_2", col]]
        print(dfcopy[col].isna().sum())

        # make sure every row is in both directions
        dfcopy = pd.concat(
            [
                dfcopy,
                dfcopy.rename(columns={"model_1": "model_2", "model_2": "model_1"}),
            ]
        )
        # make sure df is sorted
        dfcopy = dfcopy.sort_values(["model_1", "model_2"])
        print(dfcopy[col].isna().sum())

        # create pivot table
        dfpivot = dfcopy.pivot(index="model_1", columns="model_2", values=col).fillna(1)
        # print(dfpivot.isna().sum())
    else:
        dfpivot = dfcorr.copy()

    # Get the ordering from clustermap without displaying it
    if order_with_clustermap:
        # Create clustermap with plt.ioff() to prevent display
        with plt.ioff():
            g = sns.clustermap(dfpivot)

        # Get the ordering
        row_order = g.dendrogram_row.reordered_ind
        # Close the figure to free memory
        plt.close(g.fig)

        dftoplot = dfpivot.iloc[row_order, row_order]
    else:
        dftoplot = dfpivot

    # plot the heatmap
    plt.figure(figsize=(36, 30))
    # sns.heatmap(dfpivot, annot=False, cmap='coolwarm', center=1./3)

    # I only want to plot the lower triangle
    if do_mask:
        mask = np.triu(np.ones_like(dftoplot, dtype=bool))
    else:
        mask = None
    g = sns.heatmap(
        dftoplot,
        mask=mask,
        annot=False,
        cmap="coolwarm",
        cbar_kws={"label": col},
        #      'labelsize': fontsize}
    )  # , center = .8)
    # print(dftoplot)
    keyword_colors = {
        "Mistral": "blue",
        "Llama": "green",
        "Claude": "red",
        "Gpt": "purple",
        "Nova": "orange",
    }
    xticklabels = g.get_xticklabels()
    yticklabels = g.get_yticklabels()

    for label in xticklabels:
        text = label.get_text()
        for keyword, color in keyword_colors.items():
            if keyword in text:
                label.set_color(color)
                break

    for label in yticklabels:
        text = label.get_text()
        for keyword, color in keyword_colors.items():
            if keyword in text:
                label.set_color(color)
                break

    # Modify axis labels
    plt.setp(xticklabels, rotation=90, ha="right", fontsize=fontsize)
    plt.setp(yticklabels, rotation=0, ha="right", fontsize=fontsize)

    # Get the colorbar and modify its properties
    cbar = g.collections[0].colorbar
    # Modify colorbar label size
    cbar.ax.tick_params(
        labelsize=max(fontsize, 30)
    )  # Adjust the number 12 to change font size
    # If you want to add a label to the colorbar:
    colpretty = col.replace("_", " ").title()
    cbar.set_label(
        colpretty, size=max(fontsize, 30)
    )  # Adjust the number 14 to change label size

    plt.xlabel("")
    plt.ylabel("")
    # Adjust the layout to prevent label cutoff
    plt.tight_layout()

    if save:
        plt.savefig(
            f"../final_figures/4.4_{savename}heatmap_{col}.pdf",
            bbox_inches="tight",
            dpi=500,
        )

    plt.show()

    # Adjust layout to prevent label cutoff
    # g.fig.subplots_adjust(bottom=0.2, right=0.8)
