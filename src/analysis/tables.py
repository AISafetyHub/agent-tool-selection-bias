"""Generate LaTeX tables from aggregated metrics."""


def main_table(results: dict, k_values: list[int] = None) -> str:
    """Generate the main results table: Model x HSR@k + avg PED."""
    k_values = k_values or [1, 3, 5]
    overall = results["overall"]

    cols = " & ".join([f"HSR@{k}" for k in k_values] + ["PED"])
    header = f"Model & {cols} \\\\"

    rows = []
    for model, data in sorted(overall.items()):
        hsr_vals = [f"{data['hsr'].get(k, 0.0):.1%}" for k in k_values]
        ped_val = f"{data['ped']['mean']:.2f}"
        row = f"{model} & {' & '.join(hsr_vals)} & {ped_val} \\\\"
        rows.append(row)

    table = f"""\\begin{{table}}[t]
\\centering
\\begin{{tabular}}{{l{'c' * (len(k_values) + 1)}}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{Overall HSR@k and mean PED across models.}}
\\label{{tab:main_results}}
\\end{{table}}"""
    return table


def domain_table(results: dict, k: int = 5) -> str:
    """Generate breakdown table: Model x Domain HSR@k."""
    by_domain = results["by_domain"]
    domains = sorted({d for model_data in by_domain.values() for d in model_data})

    header = "Model & " + " & ".join(domains) + " \\\\"
    rows = []
    for model in sorted(by_domain):
        vals = []
        for d in domains:
            hsr = by_domain[model].get(d, {}).get("hsr", {}).get(k, 0.0)
            vals.append(f"{hsr:.1%}")
        rows.append(f"{model} & {' & '.join(vals)} \\\\")

    table = f"""\\begin{{table}}[t]
\\centering
\\small
\\begin{{tabular}}{{l{'c' * len(domains)}}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{HSR@{k} by domain.}}
\\label{{tab:domain_results}}
\\end{{table}}"""
    return table


def type_table(results: dict, k: int = 5) -> str:
    """Generate breakdown table: Model x Escalation Type HSR@k."""
    by_type = results["by_type"]
    types = sorted({t for model_data in by_type.values() for t in model_data})

    header = "Model & " + " & ".join(types) + " \\\\"
    rows = []
    for model in sorted(by_type):
        vals = []
        for t in types:
            hsr = by_type[model].get(t, {}).get("hsr", {}).get(k, 0.0)
            vals.append(f"{hsr:.1%}")
        rows.append(f"{model} & {' & '.join(vals)} \\\\")

    table = f"""\\begin{{table}}[t]
\\centering
\\small
\\begin{{tabular}}{{l{'c' * len(types)}}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{HSR@{k} by escalation-risk type.}}
\\label{{tab:type_results}}
\\end{{table}}"""
    return table
