# Synthetic demo datasets for reconciliation app

## Variance highlights
- `1000-AR` July variance 200.00 (GL 125,000 vs subledger 124,800)
- `2000-AP` August variance 600.00 (GL 102,350 vs subledger 101,750)
- `2500-AccruedExp` variances 300.00 (July) and 250.00 (August)
- `4000-Prepaid` August variance -250.00 (subledger higher than GL)
- `3000-Cash` accounts reconcile exactly

## File descriptions
- `gl_balances.csv` – month-end GL balances by account
- `subledger_balances.csv` – supporting balances per subledger report
- `transactions.csv` – optional activity used for variance explanation / roll-forward

All files use headers compatible with the Streamlit uploader. Period formatted as `YYYY-MM`, amounts in dollars.
