"""Generate LaTeX tables from aggregated metrics."""


def main_table(results: dict, k_values: list[int] = None) -> str:
    """Generate the main results table: Model x OPUR@k + avg PED."""
    k_values = k_values or [1, 3, 5]
    overall = results["overall"]

    cols = " & ".join([f"OPUR@{k}" for k in k_values] + ["PED"])
    header = f"Model & {cols} \\\\"

    rows = []
    for model, data in sorted(overall.items()):
        opur_vals = [f"{data['opur'].get(k, 0.0):.1%}" for k in k_values]
        ped_val = f"{data['ped']['mean']:.2f}"
        row = f"{model} & {' & '.join(opur_vals)} & {ped_val} \\\\"
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
\\caption{{Overall OPUR@k and mean PED across models.}}
\\label{{tab:main_results}}
\\end{{table}}"""
    return table


def domain_table(results: dict, k: int = 5) -> str:
    """Generate breakdown table: Model x Domain OPUR@k."""
    by_domain = results["by_domain"]
    domains = sorted({d for model_data in by_domain.values() for d in model_data})

    header = "Model & " + " & ".join(domains) + " \\\\"
    rows = []
    for model in sorted(by_domain):
        vals = []
        for d in domains:
            opur = by_domain[model].get(d, {}).get("opur", {}).get(k, 0.0)
            vals.append(f"{opur:.1%}")
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
\\caption{{OPUR@{k} by domain.}}
\\label{{tab:domain_results}}
\\end{{table}}"""
    return table


def type_table(results: dict, k: int = 5) -> str:
    """Generate breakdown table: Model x Escalation Type OPUR@k."""
    by_type = results["by_type"]
    types = sorted({t for model_data in by_type.values() for t in model_data})

    header = "Model & " + " & ".join(types) + " \\\\"
    rows = []
    for model in sorted(by_type):
        vals = []
        for t in types:
            opur = by_type[model].get(t, {}).get("opur", {}).get(k, 0.0)
            vals.append(f"{opur:.1%}")
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
\\caption{{OPUR@{k} by escalation-risk type.}}
\\label{{tab:type_results}}
\\end{{table}}"""
    return table
