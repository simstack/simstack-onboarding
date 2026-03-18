
@node(parameters=Parameters(force_rerun=True))
async def test_chart_generation(sample_data :PandasModel, **kwargs):
    """
    Simstack node to test chart generation with data from sample_data.csv.
    """
    node_runner = kwargs.get("node_runner")
    # csv_path = Path("private/sample_data.csv")
    # if not csv_path.exists():
    #     return node_runner.fail(f"CSV file not found: {csv_path}")

    df = sample_data.table
    # The CSV has columns: "tree_depth","n_estimators","R2_average"

    depths = sorted(df["tree_depth"].unique())
    estimators = sorted(df["n_estimators"].unique())

    # Create Plots
    # Plot 1: R2 vs tree_depth (for each n_estimators)
    depth_data_map = {}
    for _, row in df.iterrows():
        d = int(row["tree_depth"])
        if d not in depth_data_map:
            depth_data_map[d] = {"depth": d}
        depth_data_map[d][f"r2_est_{int(row['n_estimators'])}"] = float(row["R2_average"])

    pprint.pprint(depth_data_map)

    depth_all_data = sorted(depth_data_map.values(), key=lambda x: x["depth"])
    depth_series = [
        AGScatterSeriesConfig(xKey="depth", yKey=f"r2_est_{est}", title=f"estimators={est}")
        for est in estimators
    ]

    depth_chart = ChartArtifactModel(
        title=AGChartTitleConfig(text="R2 vs Tree Depth (Sample)"),
        data=depth_all_data,
        series=depth_series,
        axes=[
            AGChartAxisConfig(type="number", position="bottom", title="Tree Depth"),
            AGChartAxisConfig(type="number", position="left", title="Average R2"),
        ]
    )

    # Plot 2: R2 vs n_estimators (for each tree_depth)
    est_data_map = {}
    for _, row in df.iterrows():
        e = int(row["n_estimators"])
        if e not in est_data_map:
            est_data_map[e] = {"estimators": e}
        est_data_map[e][f"r2_depth_{int(row['tree_depth'])}"] = float(row["R2_average"])

    pprint.pprint(est_data_map)
    est_all_data = sorted(est_data_map.values(), key=lambda x: x["estimators"])
    est_series = [
        AGScatterSeriesConfig(xKey="estimators", yKey=f"r2_depth_{depth}", title=f"depth={depth}")
        for depth in depths
    ]

    est_chart = ChartArtifactModel(
        title=AGChartTitleConfig(text="R2 vs Number of Estimators (Sample)"),
        data=est_all_data,
        series=est_series,
        axes=[
            AGChartAxisConfig(type="number", position="bottom", title="Number of Estimators"),
            AGChartAxisConfig(type="number", position="left", title="Average R2"),
        ]
    )

    node_runner.depth_chart = depth_chart
    node_runner.estimators_chart = est_chart

    return node_runner.succeed()

