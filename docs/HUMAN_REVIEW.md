# Human Review

Human review is mandatory for the Accounting AI workflow.

Review UI shows:

- original PDF/image preview
- purpose
- template
- extracted fields
- AI verification result
- mappings
- validation issues

Actions:

- Save edits
- Confirm mappings
- Reject
- Approve and generate Excel
- Download Excel after completion

Approval is blocked when deterministic validation has critical issues or
required mappings are still missing.

Rejection marks the job `REJECTED` and does not generate output.
