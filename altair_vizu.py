import altair as alt
import pandas as pd
import sys

game = sys.argv[1]

source_wide = pd.read_csv(f"scores/csv/{game}_mean_scores.csv")
source_wide_std = pd.read_csv(f"scores/csv/{game}_std_scores.csv")
source_wide.rename(columns={'Unnamed: 0': 'Epochs'}, inplace=True)
source_long = source_wide.melt(["Epochs"], var_name='Algorithm', value_name="Score")
source_wide_std.rename(columns={'Unnamed: 0': 'Epochs'}, inplace=True)
source_long_std = source_wide_std.melt(["Epochs"], var_name='Algorithm', value_name="Score")
source_long["Score_min"] = source_long["Score"] - source_long_std["Score"]
source_long["Score_max"] = source_long["Score"] + source_long_std["Score"]


selection = alt.selection_multi(fields=['Algorithm'], bind='legend')
type_checkbox = alt.binding_checkbox()


base = alt.Chart(source_long).encode(x='Epochs:T')

columns = source_long.Algorithm.unique()
hover = alt.selection_single(
    fields=['Epochs'], nearest=True, on='mouseover', empty='none', clear='mouseout', name="hover"
)

points = base.mark_point().encode(x='Epochs:T', y='Score:Q', color='Algorithm:N',
        opacity=alt.condition(hover, alt.value(0.3), alt.value(0)))
lines = base.mark_line()\
        .encode(x='Epochs:T', y='Score:Q',
                color=alt.Color('Algorithm:N'),
                opacity=alt.condition(~selection, alt.value(0), alt.value(1))
                ).add_selection(selection)
areas = base.mark_area()\
        .encode(x='Epochs:T', y='Score_min:Q', y2="Score_max",
                color='Algorithm:N',
                opacity=alt.condition(~selection, alt.value(0), alt.value(0.5)))

rule = base.transform_pivot(
    'Algorithm', value='Score', groupby=['Epochs']
).mark_rule().encode(
        opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
        tooltip=[alt.Tooltip(c, type='quantitative') for c in columns]
).add_selection(
    hover
)

rolling_mean_line = base.mark_line().encode(
     x='Epochs:T', y='Score:Q', color=alt.Color('Algorithm:N')).transform_window(
        rolling_mean='mean(Score)',
        frame=[-3, 0]
).encode(
    x='Epochs:T', y='rolling_mean:Q',
    opacity=alt.condition(selection & ~selection, alt.value(1), alt.value(0))
)


# areas.encoding.y.title = ""
# areas.encoding.y2.title = ""
rolling_mean_line.encoding.y.title = "Score"

complete_chart = (lines + points + rule + areas + rolling_mean_line).configure_legend(
  orient='bottom'
)

complete_chart.save(f'html_plots/{game}_mean.html', embed_options={'actions': False})

# rolling_mean_line.save(f'html_plots/{game}_rolling.html')
