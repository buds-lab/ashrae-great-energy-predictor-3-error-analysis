import numpy as np
import pandas as pd
import datetime as dt
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
import matplotlib.dates as mdates

# Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):

    """Source code for this function:\n
        https://www.kaggle.com/caesarlupum/ashrae-start-here-a-gentle-introduction
    """

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


#Function to calculate RMSLE
def RMSLE(y_real, y_pred):

    """This function calculates RMSLE.

    y_real: real values.\n
    y_pred: predicted values.\n

    Return RMSLE value
    """

    rmsle = np.sqrt(mean_squared_log_error( y_real, y_pred ))
    return rmsle


# Function to complete dates
def complete_data(df, start_date, end_date):

    """This function reindex a df to have full dates between the period detailed

        df: dataframe to be completed\n
        start_date: start date in the format YYY-MM-DD (str)\n
        end_date: end date in the format YYY-MM-DD (str)\n

        Returns a dataframe with timestamp completed and NaN in the values
    """
    
    # List of buildings
    bdgs = list(df.building_id.unique())
    # List of dates
    idx = pd.date_range(start=start_date, end=end_date)
    dates_df = pd.DataFrame({"timestamp":idx})

    dfs = []
    for bdg in bdgs:
        # Filter building
        df2 = df[df.building_id == bdg]
        dates_len = len(df2.timestamp)

        if dates_len != len(idx):
            # Merge
            df2 = pd.merge(dates_df, df2, how="left", on="timestamp")
            df2["building_id"] = bdg
            dfs.append(df2)

        else:
            dfs.append(df2)

    # Concat
    df = pd.concat(dfs)

    del(dfs,df2)

    return df


# Function to plot heatmaps grouped by 1 feature.
def plot_heatmap_1group(group,df,cols,start,end):

    """This function plots data in the df, one subplot per element in "group". Distributed in "cols" columns

    df: input dataframe with "timestamp", "buiding_id", "rmsle_scaled" and group features\n
    group: feature to divide in subplots\n
    cols: number of columns to distribute subplots\n
    start: start date in the format "YYYY-MM-DD"\n
    end: end date in the format "YYYY-MM-DD"\n

    Return figure.
    """

    group_name = list(df[group].unique())
    print(group_name)

    plots = len(group_name)
    print(plots)

    rows = int(plots/cols) if plots % 2 == 0 else int(plots/cols)+1

    fig, axes = plt.subplots(rows, cols, sharex = True, sharey=False, figsize=(20,10))
    axes = axes.flatten()

    for i,j in enumerate(group_name):

        # Filter data
        df_grouped = df[(df[group] == j) & (df.timestamp >= start) & (df.timestamp <= end)]

        # Pivot data
        pivot_df = df_grouped.pivot(columns="timestamp", index="building_id", values="rmsle_scaled")

        # Sort in descending order (sum RMSLE)
        pivot_df["sum"] = np.sum(pivot_df,axis=1).tolist()
        pivot_df = pivot_df.sort_values("sum")
        pivot_df.drop("sum",axis=1, inplace=True)

        # Get the data
        y = np.linspace(0, len(pivot_df), len(pivot_df)+1)
        #x = mdates.drange(pivot_df.columns[0], pivot_df.columns[-1] + dt.timedelta(days=2), dt.timedelta(days=1))
        x = pd.date_range(start=start, end=end)

        # Plot
        ax = axes[i]
        data = np.array(pivot_df)
        #print(f"data shape: {data.shape}")
        cmap = plt.get_cmap('YlOrRd') #plt.get_cmap('gist_rainbow')
        qmesh = ax.pcolormesh(x, y, data, cmap=cmap, rasterized=True, vmin=0, vmax=1)
        # Axis
        ax.axis('tight') 
        ax.xaxis_date() # Set up as dates
        #plt.locator_params(axis='x', nbins=24)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y')) # set date's format
        ax.set_yticklabels([]) # omit building ID on y axis
        ax.set_title(f"{j}", fontdict={'fontsize':10})
    
    if plots % 2 != 0:
        fig.delaxes(axes[-1])

    # Color bar  
    cbaxes = fig.add_axes([0.017, 0.07, 0.975, 0.01])
    cbar = fig.colorbar(qmesh, ax=ax, orientation='horizontal', cax = cbaxes)
    cbar.set_label('Min-Max scaled RMSLE')

    fig.suptitle(f"Building ID vs. Date (by {group})", y = 1.015, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    return fig


# Function to plot heatmaps grouped by 1 feature.
def plot_heatmaps_2group(meter,group1,group2,df,cols):

    """This function plots data in the "df", one plot per element in "group1", with one subplot per element in "group2". Distributed in "cols" columns.

    meter: metername, defined globally\n
    df: input dataframe with "timestamp", "buiding_id", "rmsle_scaled" and group features\n
    group1: feature to divide in plots\n
    group2: feature to divide each plot in subplots\n
    cols: number of columns to distribute subplots\n

    Return list of figures.
    """
    figs_names = [] # here will be saved the id of figure
    figs = [] # here will be saved the figures

    group1_list = list(df[group1].unique()) # site

    for group in group1_list:
        
        figs_names.append(group)
        # Filter group1 data
        df1 = df[df[group1] == group]
        # Group2
        group2_list = list(df1[group2].unique()) # primary use
        plots = len(group2_list)      
        rows = int(plots)

        # Ratios to make height of subplot dependant of number of buldings
        ratios = [len(df1[df1[group2] == x].building_id.unique()) for x in group2_list]

        # Size of figure dependant on number of plots
        size = (15,5) if plots == 1 else (15,20)

        fig, axes = plt.subplots(rows, cols, sharex = True, sharey=False, figsize=size, gridspec_kw = {"height_ratios": ratios})
        axes = axes.flatten() if plots > 1 else axes

        for i,(j,k) in enumerate(zip(group2_list,ratios)):

            # Filter data
            df_grouped = df1[df1[group2] == j]

            # Pivot data
            pivot_df = df_grouped.pivot(columns="timestamp", index="building_id", values="rmsle_scaled")

            # Sort in descending order (sum RMSLE)
            pivot_df["sum"] = np.sum(pivot_df,axis=1).tolist()
            pivot_df = pivot_df.sort_values("sum")
            pivot_df.drop("sum",axis=1, inplace=True)

            # Get the data
            y = np.linspace(0, len(pivot_df), len(pivot_df)+1)
            x = mdates.drange(pivot_df.columns[0], pivot_df.columns[-1] + dt.timedelta(days=2), dt.timedelta(days=1))

            # Plot
            ax = axes[i] if plots > 1 else axes
            data = np.array(pivot_df)
            cmap = plt.get_cmap('YlOrRd') #plt.get_cmap('gist_rainbow') 
            qmesh = ax.pcolormesh(x, y, data, cmap=cmap, rasterized=True, vmin=0, vmax=1)
            # Axis
            ax.axis('tight') 
            ax.xaxis_date() # Set up as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y')) # set date's format
            yticks = ticker.MaxNLocator(k)
            ax.yaxis.set_major_locator(yticks)
            ax.set_yticklabels([]) # omit building ID on y axis
            ax.set_title(f"{j}", fontdict={'fontsize':10})

        # Color bar  
        bottom = 0.015 if plots == 1 else 0.09
        height = 0.01 if plots == 1 else 0.0025
        cbaxes = fig.add_axes([0.25, bottom, 0.5, height])
        cbar = fig.colorbar(qmesh, ax=ax, orientation='horizontal', cax = cbaxes)
        cbar.set_label('Min-Max scaled RMSLE')

        fig.suptitle(f"{meter} meter - Site {group}", y = 1.015, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)

        figs.append(fig)

    return figs_names, figs