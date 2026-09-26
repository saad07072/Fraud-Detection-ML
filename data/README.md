# Dataset Instructions

This repository is structured for a fraud-detection workflow using a transaction-level financial dataset. The full-scale dataset is intentionally not committed to Git because it can be very large and may be subject to licensing or privacy constraints.

## Recommended dataset

Use the public or internal financial transactions dataset that matches the fraud-detection task, such as a large synthetic or institutional transaction log with the following common fields:

- `step` : time step or transaction order
- `type` : transaction category such as PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN
- `amount` : transaction amount
- `oldbalanceOrg` : sender balance before the transaction
- `newbalanceOrig` : sender balance after the transaction
- `oldbalanceDest` : receiver balance before the transaction
- `newbalanceDest` : receiver balance after the transaction
- `isFraud` : target label, where 1 indicates fraudulent activity
- `isFlaggedFraud` : indicator used for auditing; not used as a predictive feature in this workflow

## Approximate size

The base project references a large-scale dataset with roughly 6.3 million transactions and a highly imbalanced fraud rate. The repository includes only a small sample file to demonstrate the workflow and validate the notebook end-to-end.

## How to obtain the full dataset

Place the full dataset locally in the repository at:

`data/Fraud.csv`

If you have access to a larger production or benchmark dataset, save it as:

- `data/Fraud.csv`

Then run the notebook without changing the configured paths.

## Local placement

The notebook is designed to automatically look for the dataset in this order:

1. `data/Fraud.csv`
2. `data/sample.csv`

If you are using a different filename, update the path in the notebook or the environment variable before running it.

## Notes

- Do not commit the full-size CSV to Git.
- Keep the file outside the repository if the dataset is too large or restricted.
- The pipeline was built to be reproducible and portable across machines.
