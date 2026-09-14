"""Bind manuscript numbers to the verified complete replication summary."""
from build_lcfm_replication_figures import load, cell
from build_publication_figures import ROOT, sha


def main():
    report, table, _ = load()
    macros = {}
    for result in report['primary']:
        prefix = 'LCFMReach' if result['task'] == 'reach' else 'LCFMWall'
        low, high = result['family_95_interval']
        macros[prefix+'Changed'] = str(result['changed'])
        macros[prefix+'CI'] = f'{100*low:.1f}--{100*high:.1f}\\%'
        row = cell(table, result['task'], 'original', 'all_blocks_h3_donor', 'excess_reference_cost_percent')
        macros[prefix+'Cost'] = f'{row["mean"]:.2f}\\%'
        macros[prefix+'CostCI'] = f'{row.marginal_95_low:.2f}--{row.marginal_95_high:.2f}\\%'
    for metric, name, precision, suffix, horizon in (
        ('excess_reference_cost_percent', 'LCFMCostRange', 2, r'\%', None),
        ('donor_reconstruction', 'LCFMReconRange', 4, '', 6),
        ('reference_spearman', 'LCFMRankRange', 3, '', None),
        ('reference_top10_overlap', 'LCFMEliteRange', 2, '', None)):
        values = [cell(table, task, bank, 'all_blocks_h3_donor', metric, horizon)['mean']
                  for task in ('reach', 'reach-wall') for bank in ('original', 'fresh')]
        macros[name] = f'{min(values):.{precision}f}--{max(values):.{precision}f}'+suffix
    source = ROOT/'paper/data/lcfm_replication_summary.json'
    content = '% Generated from complete 200-scenario summary SHA256 '+sha(source)+'\n'
    content += ''.join('\\newcommand{\\'+key+'}{'+value+'}\n' for key, value in macros.items())
    path = ROOT/'paper/lcfm/replication_numbers.tex'
    path.write_text(content)
    print('Manuscript values bound to all 200 audited scenarios')


if __name__ == '__main__':
    main()
