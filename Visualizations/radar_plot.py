import pandas as pd
import plotly.graph_objects as go

# --- Load corrected data into DataFrame ---
data = {
    "dataset": [
        "H3","H3K14ac","H3K36me3","H3K4me1","H3K4me2","H3K4me3","H3K79me3","H3K9ac",
        "H4","H4ac","covid","mouse0","mouse1","mouse2","mouse3","mouse4",
        "prom_300_all","prom_300_notata","prom_300_tata",
        "prom_core_all","prom_core_notata","prom_core_tata",
        "splice","tf0","tf1","tf2","tf3","tf4"
    ],
    "baseline CNN": [61.52,29.73,38.6,26.06,25.76,20.52,46.3,40.03,
                     62.34,25.54,22.23,31.14,59.74,63.15,45.48,27.18,
                     75.78,85.14,70.3,
                     58.07,60.09,69.33,
                     76.79,53.95,63.2,45.22,29.84,61.48],
    "HyenaDNA": [67.17,31.98,48.27,35.83,25.81,23.15,54.09,50.84,
                 73.69,38.44,23.27,35.62,80.5,65.34,54.2,19.17,
                 47.38,52.24,5.34,
                 36.95,35.38,72.87,
                 72.67,62.3,67.86,46.85,41.78,61.23],
    "NT-500M-human": [69.67,33.55,44.14,37.15,30.87,24.06,58.35,45.81,
                      76.17,33.74,50.82,31.04,75.04,61.67,29.17,29.27,
                      87.71,90.75,78.07,
                      63.45,64.82,71.34,
                      79.71,61.59,66.75,53.58,42.95,60.81],
    "NT-500M-1000g": [72.52,39.37,45.58,40.45,31.05,26.16,59.33,49.29,
                      76.29,36.79,52.06,39.26,75.49,64.7,33.07,34.01,
                      89.76,91.75,78.23,
                      66.7,67.17,73.52,
                      80.97,63.64,70.17,52.73,45.24,62.82],
    "NT-2500M-1000g": [74.61,44.08,50.86,43.1,30.28,30.87,61.2,52.36,
                       79.76,41.46,66.73,48.31,80.02,70.14,42.25,43.4,
                       90.95,93.07,75.8,
                       67.39,67.46,69.66,
                       85.78,48.31,80.02,70.14,42.25,43.4],
    "NT-2500M-multi": [78.77,56.2,61.99,55.3,36.49,40.34,64.7,56.01,
                       81.67,49.13,73.04,63.31,83.76,71.52,69.44,47.07,
                       91.01,94.0,79.43,
                       70.33,71.58,72.97,
                       73.04,63.31,83.76,71.52,69.44,47.07],
    "GROVER": [72.54,41.11,47.52,37.82,28.33,22.08,57.7,51.34,
               74.42,37.74,68.49,53.33,75.94,82.32,72.45,41.47,
               75.47,87.81,59.72,
               61.09,65.82,74.26,
               82.87,64.52,64.06,61.51,50.13,75.99],
    "Gena-LM": [72.22,42.55,52.16,42.38,32.25,6.51,61.09,53.76,
                77.51,35.14,4,48.79,83.14,73.72,30.16,45.19,
                85.56,93.22,61.47,
                63.08,66.41,66.48,
                83.59,63.99,70.79,64.57,56.34,74.01],
    "DNABERT-2": [78.27,52.57,56.88,50.52,31.13,36.27,67.39,55.63,
                  80.71,50.43,71.02,56.26,84.77,79.32,66.47,52.66,
                  71.59,94.27,86.77,
                  74.17,68.04,69.37,
                  84.99,71.99,76.06,66.52,58.54,77.43],
    "DNABERT-2*": [80.17,57.42,61.9,53,39.89,41.2,65.46,57.07,
                    81.86,50.35,68.49,64.23,86.26,81.28,73.49,50.8,
                    68.79,94.34,88.31,
                    76.18,69.53,67.5,
                    85.93,69.12,71.87,62.96,55.35,74.94],
    "Ours_depth_control": [76,84.7,78.39,66.19,67.8,77.96,77.33,73.31,
                       77.33,79.69,72.39,50.71,70.44,73.16,56.82,38.21,
                       68.11,76.52,25.97,
                       53.91,57.58,50.14,
                       38.63,60.38,65.1,58.06,42.98,72.19],
    "Ours_no_control": [75.87,84.7,78.39,66.19,67.8,77.47,77.33,73.31,
                        77.33,79.69,72.78,50.71,70.44,73.19,56.82,38.21,
                        59.24,64.26,28.98,
                        45.34,49.26,50.14,
                        37.03,60.38,65.1,58.06,42.98,72.19]
}

df = pd.DataFrame(data)

# --- Radar plot setup ---
categories = df["dataset"].tolist()

fig = go.Figure()

color_map = {
    "baseline CNN": "darkgrey"   # make CNN grey
    # "Ours_depth_control": "blue",     # force LZ_d to blue
    # "Ours_no_control": "lightblue",
    # "DNABERT-2": "green",
    # "DNABERT-2-*": "darkgreen"
    # you can add more overrides if you want
}
# Add each model as a trace
for col in df.columns[1:]:
    fig.add_trace(go.Scatterpolar(
        r=df[col].tolist(),
        theta=categories,
        fill='toself',
        name=col,
        line=dict(color=color_map.get(col, None)),  # use custom color if defined
        visible="legendonly" if col not in ["Ours_depth_control","DNABERT-2","DNABERT-2*"] else True
    ))

# Layout
fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0,100])
    ),
    showlegend=True,
    title="Performance of gLMs and our LZ78-based Classifier on GUE Tasks"
)

import plotly.io as pio
pio.write_html(fig, file="radar_plot.html", auto_open=True)


fig.show()
# import caas_jupyter_tools
# caas_jupyter_tools.display_dataframe_to_user("Updated model performance dataframe", df)
