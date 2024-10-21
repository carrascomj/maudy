"""Test approximate posterior analysis."""

from maudy.analysis import predict, report_to_dfs
from maudy.train import train


def test_end_to_end_predict_retrieves_requested_variables(maud_input):
    var_names = (
        "y_flux_train",
        "latent_bal_conc",
        "unb_conc",
        "ssd",
        "dgr",
        "flux",
        "ln_bal_conc",
        "quench_correction",
    )
    maudy, _ = train(maud_input, 100, True, False, True, True, 10, False)
    samples = predict(maudy, 100, var_names=var_names)
    if "flux" in samples:
        samples["flux"] = samples["flux"].squeeze(1)
    df_result = report_to_dfs(maudy, samples, var_names=list(var_names))
    assert all(df_result[var_name].shape[0] != 0 for var_name in var_names)
    assert len(df_result) == len(var_names)
